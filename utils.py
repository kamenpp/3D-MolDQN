from rdkit.Chem.Draw.rdMolDraw2D import MolDraw2DSVG
from IPython.display import display, SVG
d2d = MolDraw2DSVG(250,200)
d2d.drawOptions().legendFraction = 0.18
d2d.drawOptions().legendFontSize = 18
opts = d2d.drawOptions()
d2d.FinishDrawing()

def draw_top_mols_to_grids(df, row_for_sorting):
    top_10 = [Chem.MolFromSmiles(smi) for smi in df.sort_values(row_for_sorting).tail(10)['smiles'].values]
    top_10_labels = [f'episode: {episode}, steps left: {steps_left}, reward: {round(reward, 2)} \n similarity: {round(sim,2)}, IPen: {pen}, SuCOS: {round(sucos, 2)} \n NDS: {round(nds, 2)}, NIPen: {round(nipen,2)}, EP: {round(env_pen, 2)} \n SAS: {round(sascore, 2)}, LE: {round(le,2)}, EP+NDS: {round(env_penalty_and_neg_docking_score,2)}, PNDS: {round(pen_neg_doc,2)}' 
                     for episode, steps_left, reward,sim, pen, sucos, nds,nipen,env_pen,sascore,le, env_penalty_and_neg_docking_score, pen_neg_doc
                                    in zip(df.sort_values(row_for_sorting).tail(10)['episode'].values, df.sort_values(row_for_sorting).tail(10)['steps_left'].values, df.sort_values(row_for_sorting).tail(10)['reward'].values,
                                           df.sort_values(row_for_sorting).tail(10)['similarity'].values, df.sort_values(row_for_sorting).tail(10)['interactions_penalty'].values,
                                          df.sort_values(row_for_sorting).tail(10)['sucos'].values, df.sort_values(row_for_sorting).tail(10)['negative_docking_score'].values,
                                           df.sort_values(row_for_sorting).tail(10)['normalised_interactions_penalty'].values,df.sort_values(row_for_sorting).tail(10)['penalty'].values,
                                           df.sort_values(row_for_sorting).tail(10)['sascore'].values,df.sort_values(row_for_sorting).tail(10)['ligand_efficiency'].values, df.sort_values(row_for_sorting).tail(10)['docking_env_penalty'].values,
                                            df.sort_values(row_for_sorting).tail(10)['penalised_negative_docking_score'].values) ]
    return Draw.MolsToGridImage(top_10, legends=top_10_labels, useSVG=True, molsPerRow=5, subImgSize=(300,300), drawOptions=opts, returnPNG=False)


def separate_not_failed_runs(df_all, df_last_steps):
    not_failed_df_all = df_all.loc[df_all['sucos'] > 0.]
    failed_df_all = df_all.loc[df_all['sucos'] == 0.]
    not_failed_df_last_steps = not_failed_df_all.loc[not_failed_df_all['steps_left']==1]
    return not_failed_df_all, not_failed_df_last_steps, failed_df_all

def floatify(x):
    try:
        return float(x)
    except:
        print(x)
        return float('nan')
    
def cumulative_reward(df_all, baseline):
    dep_baseline = baseline
    cum_dep_baseline = 0.
    for i in range(10):
        cum_dep_baseline += dep_baseline * (0.9 ** i)

    df_all['cum_reward'] = df_all.groupby('episode')['discounted_reward'].cumsum()

    df_all['reward_improvement'] = (df_all['reward'] - dep_baseline) /  dep_baseline
    df_all['cum_reward_improvement'] = (df_all['cum_reward'] - cum_dep_baseline) / cum_dep_baseline

def draw_top_scaffolds_to_grids(df, row_for_sorting):
    top_10 = [Chem.MolFromSmiles(smi) for smi in df.sort_values(row_for_sorting).tail(10)['framework'].values]
    top_10_labels = [f'episode: {episode}, steps left: {steps_left} \n similarity: {round(sim,2)}, IPen: {pen}, SuCOS: {round(sucos, 2)} \n NDS: {nds}, NIPen: {round(nipen,2)}, EP: {round(env_pen, 2)} \n SAS: {round(sascore, 2)}, LE: {round(le,2)}, EP+NDS: {round(env_penalty_and_neg_docking_score,2)}, PNDS: {round(pen_neg_doc,2)}' 
                     for episode, steps_left, sim, pen, sucos, nds,nipen,env_pen,sascore,le, env_penalty_and_neg_docking_score, pen_neg_doc
                                    in zip(df.sort_values(row_for_sorting).tail(10)['episode'].values, df.sort_values(row_for_sorting).tail(10)['steps_left'].values,
                                           df.sort_values(row_for_sorting).tail(10)['similarity'].values, df.sort_values(row_for_sorting).tail(10)['interactions_penalty'].values,
                                          df.sort_values(row_for_sorting).tail(10)['sucos'].values, df.sort_values(row_for_sorting).tail(10)['negative_docking_score'].values,
                                           df.sort_values(row_for_sorting).tail(10)['normalised_interactions_penalty'].values,df.sort_values(row_for_sorting).tail(10)['penalty'].values,
                                           df.sort_values(row_for_sorting).tail(10)['sascore'].values,df.sort_values(row_for_sorting).tail(10)['ligand_efficiency'].values, df.sort_values(row_for_sorting).tail(10)['docking_env_penalty'].values,
                                            df.sort_values(row_for_sorting).tail(10)['penalised_negative_docking_score'].values)]
    return Draw.MolsToGridImage(top_10, legends=top_10_labels, useSVG=True, molsPerRow=5, subImgSize=(300,300), drawOptions=opts)

def get_episode_category_and_last_steps(df_all):
    df_all['episode_category'] = pd.cut(df_all['episode'], bins=[-1,49,99,149,199,249,300], 
                                                  labels=['0-50','50-100','100-150','150-200','200-250','250-300'])
    df_last_steps = df_all.loc[df_all['steps_left']==1]
    return df_last_steps

def average_agg_tanimoto(stock_vecs, gen_vecs,
                         batch_size=3000, agg='max',
                         device='cpu', p=1):
    """
    For each molecule in gen_vecs finds closest molecule in stock_vecs.
    Returns average tanimoto score for between these molecules
    Parameters:
        stock_vecs: numpy array <n_vectors x dim>
        gen_vecs: numpy array <n_vectors' x dim>
        agg: max or mean
        p: power for averaging: (mean x^p)^(1/p)
    """
    assert agg in ['max', 'mean'], "Can aggregate only max or mean"
    agg_tanimoto = np.zeros(len(gen_vecs))
    total = np.zeros(len(gen_vecs))
    for j in range(0, stock_vecs.shape[0], batch_size):
        x_stock = torch.tensor(stock_vecs[j:j + batch_size]).to(device).float()
        for i in range(0, gen_vecs.shape[0], batch_size):
            y_gen = torch.tensor(gen_vecs[i:i + batch_size]).to(device).float()
            y_gen = y_gen.transpose(0, 1)
            tp = torch.mm(x_stock, y_gen)
            jac = (tp / (x_stock.sum(1, keepdim=True) +
                         y_gen.sum(0, keepdim=True) - tp)).cpu().numpy()
            jac[np.isnan(jac)] = 1
            if p != 1:
                jac = jac**p
            if agg == 'max':
                agg_tanimoto[i:i + y_gen.shape[1]] = np.maximum(
                    agg_tanimoto[i:i + y_gen.shape[1]], jac.max(0))
            elif agg == 'mean':
                agg_tanimoto[i:i + y_gen.shape[1]] += jac.sum(0)
                total[i:i + y_gen.shape[1]] += jac.shape[0]
    if agg == 'mean':
        agg_tanimoto /= total
    if p != 1:
        agg_tanimoto = (agg_tanimoto)**(1/p)
    return np.mean(agg_tanimoto)

def internal_diversity(gen, n_jobs=1, device='cpu', fp_type='morgan',
                       gen_fps=None, p=1):
    """
    Computes internal diversity as:
    1/|A|^2 sum_{x, y in AxA} (1-tanimoto(x, y))
    """
    if gen_fps is None:
        gen_fps = np.array([AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), radius=2, useChirality=True)for smi in gen]) # fingerprints(gen, fp_type=fp_type, n_jobs=n_jobs)
    return 1 - (average_agg_tanimoto(gen_fps, gen_fps,
                                     agg='mean', p=p)).mean()


def get_cumulative_diversity(dep_last_steps):
    cum_diversity_dep_last_steps_atoms = []
    for i in range(50, 301, 50):
        cum_diversity_dep_last_steps_atoms.append(internal_diversity(dep_last_steps.iloc[:i].smiles))
    return cum_diversity_dep_last_steps_atoms


from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmilesFromSmiles

def get_unique_scaffolds(plif_all):
    plif_all['framework'] = plif_all.smiles.apply(MurckoScaffoldSmilesFromSmiles)

    scaffold_df_plif = plif_all.framework.value_counts().reset_index().copy() # cool trick
    scaffold_df_plif.columns = ["scaffold","count"]
    # copy the index column for the dataframe to scaffold_idx
    scaffold_df_plif['scaffold_idx'] = scaffold_df_plif.index

    scaffold_dict_plif = dict(zip(scaffold_df_plif.scaffold,scaffold_df_plif.index))
    plif_all['scaffold_idx'] = [scaffold_dict_plif[x] for x in plif_all.framework]

    scaffold_df_plif['mol'] = scaffold_df_plif['scaffold'].apply(Chem.MolFromSmiles)
    crds_ok = scaffold_df_plif.mol.apply(AllChem.Compute2DCoords)

    plif_all['mol'] = plif_all.smiles.apply(Chem.MolFromSmiles)
    for idx, scaf in scaffold_df_plif[["scaffold_idx","mol"]].values:
        match_df = plif_all.query("scaffold_idx == @idx")
        for mol in match_df.mol:
            try:
                AllChem.GenerateDepictionMatching2DStructure(mol,scaf)
            except Exception as e:
                print(repr(e))

    scaffold_count_dict_plif = dict(zip(scaffold_df_plif.scaffold,scaffold_df_plif['count']))
    inverse_scaffold_dict_plif = dict((v,k) for k, v in scaffold_dict_plif.items())
    plif_all['scaffold_count'] = [scaffold_count_dict_plif[x] for x in plif_all.framework]
    unique_scaffolds_plif = plif_all.groupby('scaffold_idx').max('interactions_penalty')
    unique_scaffolds_plif['framework'] = [inverse_scaffold_dict_plif[ix] for ix in unique_scaffolds_plif.index]
    
    return unique_scaffolds_plif

def draw_scaffolds_to_grids(df, row_for_sorting, how_many):
    top_10 = [Chem.MolFromSmiles(smi) for smi in df.sort_values(row_for_sorting).tail(how_many)['framework'].values]
    top_10_labels = [f'Count: {count}, IPen: {round(reward, 2)}'
                     for count, reward
                                    in zip(df.sort_values(row_for_sorting).tail(how_many)['scaffold_count'].values, df.sort_values(row_for_sorting).tail(how_many)['interactions_penalty'].values)]
    return Draw.MolsToGridImage(top_10, legends=top_10_labels, useSVG=True, molsPerRow=5, subImgSize=(300,300), drawOptions=opts)

class SillyWalks:
    def __init__(self, df):
        self.count_dict = {}
        for smi in df.SMILES:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fp = AllChem.GetMorganFingerprint(mol, 2, useChirality=True)
                for k, v in fp.GetNonzeroElements().items():
                    self.count_dict[k] = self.count_dict.get(k, 0) + v

    def score(self, smiles_in):
        mol = Chem.MolFromSmiles(smiles_in)
        if mol:
            fp = AllChem.GetMorganFingerprint(mol, 2)
            on_bits = fp.GetNonzeroElements().keys()
            silly_bits = [
                x for x in [self.count_dict.get(x) for x in on_bits] if x is None
            ]
            score = len(silly_bits) / len(on_bits)
        else:
            score = 1
        return score
ref_df = pd.read_csv("/Users/kamen/Dev/silly_walks/chembl_drugs.smi", sep=" ", names=['SMILES', 'Name'])
silly_walks = SillyWalks(ref_df)

from sklearn.preprocessing import MinMaxScaler
mms=MinMaxScaler()

def get_top_10_avgs(dep_all):
    """
    sn.lineplot(x=range(len(top10_avgs)),y=top10_avgs) to plot
    """
    rewards = np.where(dep_all['reward'].values < 0., 0., dep_all['reward'].values)

    scaled_rewards = mms.fit_transform(rewards.reshape(-1, 1) )
    scaled_rewards = scaled_rewards.reshape(-1)
    
    top10_avgs = [0] * 10
    for i in range(10, 3000):
        rewards = scaled_rewards[:i].copy()
        rewards = np.sort(rewards)
        top10_avgs.append(np.mean(rewards[-10:]))
    
    return top10_avgs


def plt_sascore_scatter(failed_df_all, not_failed_df_all, df_all, title, save_path):
    df_all['mean_sascore'] = df_all.groupby(['episode_category'])['sascore'].transform('mean')
    not_failed_df_all['mean_sascore'] = not_failed_df_all.groupby(['episode_category'])['sascore'].transform('mean')
    failed_df_all['mean_sascore'] = failed_df_all.groupby(['episode_category'])['sascore'].transform('mean')
    fig = figure(figsize=(16,9))
    ax = plt.subplot(1,1,1)
    sn.scatterplot(data=failed_df_all,x='episode',y='sascore', ax=ax, s=5, color='blue', label='failures')
    graph = sn.scatterplot(data=not_failed_df_all,x='episode',y='sascore', ax=ax,s=5, color='green', label='non-failures')
    sn.lineplot(data=not_failed_df_all,x='episode',y='mean_sascore', ax=ax, color='green', label='mean SAScore of non-failures over episode category')
    sn.lineplot(data=failed_df_all,x='episode',y='mean_sascore', ax=ax, color='blue', label='mean SAScore of failures over episode category')
    sn.lineplot(data=df_all,x='episode',y='mean_sascore', ax=ax, color='magenta', label='mean SAScore of all over episode category')
    graph.axhline(3.59, color='red', label='troglitazone SAScore')
    plt.title(title)
    plt.ylabel('SAScore')
    plt.xlabel('Episode')
    plt.legend()
    plt.tight_layout()

    
def plt_num_hd_scatter(failed_df_all, not_failed_df_all, df_all, title):
    df_all['mean_hd'] = df_all.groupby(['episode_category'])['num_hd'].transform('mean')
    not_failed_df_all['mean_hd'] = not_failed_df_all.groupby(['episode_category'])['num_hd'].transform('mean')
    failed_df_all['mean_hd'] = failed_df_all.groupby(['episode_category'])['num_hd'].transform('mean')
    fig = figure(figsize=(16,9))
    ax = plt.subplot(1,1,1)
    sn.scatterplot(data=failed_df_all,x='episode',y='num_hd', ax=ax, s=5, color='blue', label='failures')
    graph = sn.scatterplot(data=not_failed_df_all,x='episode',y='num_hd', ax=ax,s=5, color='green', label='non-failures')
    sn.lineplot(data=not_failed_df_all,x='episode',y='mean_hd', ax=ax, color='green', label='mean number of HD of non-failures over episode category')
    sn.lineplot(data=failed_df_all,x='episode',y='mean_hd', ax=ax, color='blue', label='mean number of HD of failures over episode category')
    sn.lineplot(data=df_all,x='episode',y='mean_hd', ax=ax, color='magenta', label='mean number of HD of all over episode category')
    graph.axhline(1.0, color='red', label='celecoxib number of HD')
    plt.title(title)
    plt.ylabel('Number of HD')
    plt.xlabel('Episode')
    plt.legend()
    plt.tight_layout()
    
def plt_num_ha_scatter(failed_df_all, not_failed_df_all, df_all, title):
    df_all['mean_ha'] = df_all.groupby(['episode_category'])['num_ha'].transform('mean')
    not_failed_df_all['mean_ha'] = not_failed_df_all.groupby(['episode_category'])['num_ha'].transform('mean')
    failed_df_all['mean_ha'] = failed_df_all.groupby(['episode_category'])['num_ha'].transform('mean')
    fig = figure(figsize=(16,9))
    ax = plt.subplot(1,1,1)
    sn.scatterplot(data=failed_df_all,x='episode',y='num_ha', ax=ax, s=5, color='blue', label='failures')
    graph = sn.scatterplot(data=not_failed_df_all,x='episode',y='num_ha', ax=ax,s=5, color='green', label='non-failures')
    sn.lineplot(data=not_failed_df_all,x='episode',y='mean_ha', ax=ax, color='green', label='mean number of HA of non-failures over episode category')
    sn.lineplot(data=failed_df_all,x='episode',y='mean_ha', ax=ax, color='blue', label='mean number of HA of failures over episode category')
    sn.lineplot(data=df_all,x='episode',y='mean_ha', ax=ax, color='magenta', label='mean number of HA of all over episode category')
    graph.axhline(4.0, color='red', label='celecoxib number of HA')
    plt.title(title)
    plt.ylabel('Number of HA')
    plt.xlabel('Episode')
    plt.legend()
    plt.tight_layout()
    
    
def draw_similarity_uniques(unique_mols, df_last_steps, top_n):
    top = unique_mols.sort_values('interactions_penalty').tail(top_n)['smiles'].values
    top_fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi),radius=3, useChirality=True) for smi in top]

    results = []
    for fp in top_fps:
        res = np.stack(np.array([round(DataStructs.TanimotoSimilarity(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi),radius=3, useChirality=True), fp),2)              for smi in df_last_steps['smiles'].values]), axis=0)
        results.append(res)

    plt.figure(figsize=(30, 15))
    sn.heatmap(np.array(results, dtype=np.float32), yticklabels=[f'top {str(i)}' for i in range(top_n, 0, -1)]) #
    plt.xlabel('Episode', fontsize = 20)
    plt.ylabel('Unique molecules of highest reward', fontsize = 20)
    plt.title(f'Tanimoto similarity (ECFP6) of the molecules from the end of episodes to the top {top_n} unique molecules', fontsize = 20)
    plt.tight_layout()
    

def draw_bottom_mols_to_grids(df, row_for_sorting):
    top_10 = [Chem.MolFromSmiles(smi) for smi in df.sort_values(row_for_sorting).head(10)['smiles'].values]
    top_10_labels = [f'episode: {episode}, steps left: {steps_left} \n similarity: {round(sim,2)}, IPen: {pen}, SuCOS: {round(sucos, 2)} \n NDS: {nds}, NIPen: {round(nipen,2)}, EP: {round(env_pen, 2)} \n SAS: {round(sascore, 2)}, LE: {round(le,2)}, EP+NDS: {round(env_penalty_and_neg_docking_score,2)}, PNDS: {round(pen_neg_doc,2)}' 
                     for episode, steps_left, sim, pen, sucos, nds,nipen,env_pen,sascore,le, env_penalty_and_neg_docking_score, pen_neg_doc
                                    in zip(df.sort_values(row_for_sorting).head(10)['episode'].values, df.sort_values(row_for_sorting).head(10)['steps_left'].values,
                                           df.sort_values(row_for_sorting).head(10)['similarity'].values, df.sort_values(row_for_sorting).head(10)['interactions_penalty'].values,
                                          df.sort_values(row_for_sorting).head(10)['sucos'].values, df.sort_values(row_for_sorting).head(10)['negative_docking_score'].values,
                                           df.sort_values(row_for_sorting).head(10)['normalised_interactions_penalty'].values,df.sort_values(row_for_sorting).head(10)['penalty'].values,
                                           df.sort_values(row_for_sorting).head(10)['sascore'].values,df.sort_values(row_for_sorting).head(10)['ligand_efficiency'].values, df.sort_values(row_for_sorting).head(10)['docking_env_penalty'].values,
                                            df.sort_values(row_for_sorting).head(10)['penalised_negative_docking_score'].values)]
    return Draw.MolsToGridImage(top_10, legends=top_10_labels, useSVG=True, molsPerRow=5, subImgSize=(300,300), drawOptions=opts)



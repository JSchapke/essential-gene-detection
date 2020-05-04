

def essentiality_data(
        organism='', 
        ppi='biogrid', 
        expression=False, orthologs=False,
        sublocalizations=False, 
        string_params={}, seed=0, weights=False):

    cache = '.cache/yeast/'
    cachepath = cache + f'{expression}_{orthologs}_{ppi}.pkl'
    os.makedirs(cache, exist_ok=True)

    if os.path.isfile(cachepath):
        print('Data was cached')
        with open(cachepath, 'rb') as f:
            return pickle.load(f)


    edges = None
    edge_weights = None
    if ppi == 'biogrid' or ppi == 'gatplus':
        ppi_path = os.path.join(DATA_ROOT, f'essential_genes/{organism}/PPI/biogrid.csv')
        edges = pd.read_csv(ppi_path)

    if ppi == 'dip':
        ppi_path = os.path.join(DATA_ROOT, 'essential_genes/yeast/PPI/dip.csv')
        ppi = pd.read_csv(ppi_path)
        edges = ppi[['0', '1']]

    if ppi == 'string' or ppi == 'gatplus':
        ppi_path = os.path.join(DATA_ROOT, 'essential_genes/yeast/PPI/STRING/string.csv')
        string = pd.read_csv(ppi_path)
        
        key = string_params['key'] if 'key' in string_params else 'combined_score'
        string = string[['A', 'B', key]].dropna()

        if weights:
            edge_weights = string[key].values / 1000
        else:
            thr = string_params['thr']*1000 if 'thr' in string_params else 500
            string = string[string.loc[:, key] > thr]

        edges = string[['A', 'B']].values

    if edges is None:
        raise Exception('PPI dataset not supported.')

    genes = np.union1d(edges[:, 0], edges[:, 1])
    X = np.zeros((len(genes), 0))

    labels_path = os.path.join(DATA_ROOT, 'essential_genes/yeast/EssentialGenes/ogee.csv')
    labels = pd.read_csv(labels_path)

    #if ppi == 'gatplus':
    #    bioG = nx.Graph()
    #    bioG.add_edges_from(bio.values)
    #    stringG = nx.Graph()
    #    stringG.add_edges_from(string.values)
    #    G = [bioG, stringG]
    #else:
    #    G = nx.Graph()
    #    G.add_edges_from(ppi.values)

    if expression:
        path = os.path.join(DATA_ROOT, 'essential_genes/yeast/Expression/Filtered.csv')
        profile = pd.read_csv(path)
        values = profile[profile.columns[1:]].values
        #genes = profile['Gene'].values.astype(str)

        x = np.zeros((len(genes), profile.shape[1]-1))
        #X = []
        #for node in list():
        #    if node not in nodes:
        #        G.remove_node(node)
        #    else:
        #        X.append(values[genes==node].squeeze())

        for i, gene in enumerate(genes):
            mask = profile['Gene'].values == gene
            if np.any(mask):
                x[i] = profile[mask].values[0, 1:]
        X = np.concatenate([X, x], axis=1)


        #X = np.array(X, dtype=np.float32)
        print('Gene expression dataset shape:', profile.shape)

    if orthologs:
        path = os.path.join(DATA_ROOT, 'essential_genes/yeast/Orthologs/Orthologs.csv')
        orths = pd.read_csv(path)
        print('Orthologs dataset shape:', orths.shape)

        x = np.zeros((len(genes), orths.shape[1]-1))

        for i, node in enumerate(genes):
            mask = orths['Gene'] == node
            if np.any(mask):
                x[i] = orths[mask].values.squeeze()[:-1]
        X = np.concatenate([X, np.array(x)], axis=1)

    if sublocalizations:
        #path = os.path.join(DATA_ROOT, 'essential_genes/yeast/SubLocalizations/SubLocalizations.npy')
        path = os.path.join(DATA_ROOT, 'essential_genes/yeast/SubLocalizations/Localizations11.npy')
        subloc = np.load(path, allow_pickle=True)
        print('Subcellular Localizations dataset shape:', subloc.shape)

        x = np.zeros((len(genes), subloc.shape[1]-1))

        for i, gene in enumerate(genes):
            mask = subloc[:, 0] == gene
            if np.any(mask):
                x[i] = subloc[mask, 1:].astype(float).max(0)
        X = np.concatenate([X, np.array(x)], axis=1)

    idx = []
    for i, label in labels.iterrows():
        if label['Gene'] in genes:
            idx.append(i)
    labels = labels.iloc[idx].values

    train, test = train_test_split(labels, test_size=0.2, random_state=seed, stratify=labels[:, 1])

    #with open(cachepath, 'wb') as f:
    #    pickle.dump([G, X, train, test], f, protocol=2)

    #print(f'G: n.nodes ({len(G.nodes)}) , n.edges ({len(G.edges)})')
    print(f'X.shape: {None if X is None else X.shape}.')
    print('Train labels:', len(train))
    print('Test labels:', len(test))

    return (edges, edge_weights), X, train, test

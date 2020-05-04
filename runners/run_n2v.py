def n2v(X, A, train_y, train_mask, val_y, val_mask):
    from torch_geometric.nn import Node2Vec
    n2v_path = 'models/node2vec/node2vec.pickle'
    params = { 
            'embedding_dim': 32,
            'walk_length': 64,
            'context_size': 64,
            'walks_per_node': 2 } 

    def train(embedding_dim, walk_length, context_size, walks_per_node=1):
        print(embedding_dim, walk_length, context_size, walks_per_node)
        model = Node2Vec(len(X), embedding_dim, walk_length, context_size, walks_per_node)

        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        nodes = range(len(X))
        subsample_size = 100

        iterable = tqdm(range(5000)) 
        loss_avg = deque(maxlen=100)
        for i in iterable:
            subsample = random.sample(nodes, subsample_size)
            subsample = torch.tensor(subsample)

            loss = model.loss(A, subsample)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_avg.append(loss)
            tqdm.set_description(iterable, desc='Loss: %.4f' % (loss), refresh=True)
        return model, sum(loss_avg) / 100

    def objective(trial):
        embedding_dim = 2 ** trial.suggest_int('embedding_dim', 2, 8)
        walk_length = 2 ** trial.suggest_int('walk_length', 0, 6)
        context_size = min(2**trial.suggest_int('context_size', 0, 6), walk_length)
        walks_per_node = trial.suggest_int('walks_per_node', 1, 8)

        _, score = train(embedding_dim, walk_length, context_size, walks_per_node)
        return score
        
    def train_SVM():
        from sklearn.svm import SVC
        svm = SVC()

        N2V = Node2Vec(len(X), **params)
        N2V.load_state_dict(torch.load(n2v_path))
        N2V.eval()

        train_X = np.where(train_mask)[0]
        train_X = torch.tensor(train_X, dtype=int)
        with torch.no_grad():
            Z = N2V.forward(train_X).numpy()
        svm.fit(Z, train_y)
        train_preds = svm.predict(Z)
        print('Train Accuracy:')
        print(np.sum(train_preds == train_y) / len(train_y))

        val_X = np.where(val_mask)[0]
        val_X = torch.tensor(val_X, dtype=int)
        with torch.no_grad():
            Z = N2V.forward(val_X).numpy()
        val_preds = svm.predict(Z)
        print('Val Accuracy:')
        print(np.sum(val_preds == val_y) / len(val_y))
        print(val_preds)

    #study = optuna.create_study()
    #study.optimize(objective, timeout=60*60)

    #model, score = train(**params)
    #torch.save(model.state_dict(), n2v_path)

    train_SVM()


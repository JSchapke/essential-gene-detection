from torch_geometric.nn import Node2Vec
from sklearn.svm import SVC
from torch.nn import optim
from tqdm import tqdm

params = { 
        'embedding_dim': 32,
        'walk_length': 64,
        'context_size': 64,
        'walks_per_node': 2 } 

class Node2Vec:
    def __init__(self, A):
        self.n2v = Node2Vec(len(A), **params)
        self.svm = SVC()

    def fit(train_mask, train_y):
        n2v = self.n2v
        svm = self.svm
   
        n2v.train()
        optimizer = optim.Adam(n2v.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        nodes = range(len(X))
        subsample_size = 100

        iterable = tqdm(range(5000)) 
        loss_avg = deque(maxlen=100)
        for i in iterable:
            subsample = random.sample(nodes, subsample_size)
            subsample = torch.tensor(subsample)

            loss = n2v.loss(A, subsample)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_avg.append(loss)
            tqdm.set_description(iterable, desc='Loss: %.4f' % (loss), refresh=True)

        train_X = np.where(train_mask)[0]
        train_X = torch.tensor(train_X, dtype=int)

        n2v.eval()
        with torch.no_grad():
            Z = n2v.forward(train_X).numpy()
        svm.fit(Z, train_y)
        print('Finished Training SVM.')


    def predict(node_mask):
        X = np.where(train_mask)[0]
        X = torch.tensor(train_X, dtype=int)

        self.n2v.eval()
        with torch.no_grad():
            Z = self.n2v.forward(train_X).numpy()
        preds = self.svm.predict(Z)
        return preds





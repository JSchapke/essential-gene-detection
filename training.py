import torch
from utils import evalAUC

def train(model, optmizer, loss_fnc, X, A, 
        train_y, train_idx, val_y, val_idx, epochs=1000, early_stop_patience=20,
        return_score=False, savepath='', device=torch.device('cuda')):

    X, A = X.to(device), A.to(device)
    train_y = train_y.to(device)
    val_y = val_y.to(device)

    stop_steps = best_auc = 0
    val_steps = 100

    iterable = tqdm(epochs) 
    for i in iterable:
        model.train()
        logits = model(X, A)

        idxs = torch.tensor(train_idx)
        positive_idxs = idxs[train_y == 1]
        negative_idxs = idxs[train_y == 0]

        positives = train_y[train_y == 1]
        negatives = train_y[train_y == 0]

        loss_pos = loss_fnc(logits[positive_idxs].squeeze(), positives)
        loss_neg = loss_fnc(logits[negative_idxs].squeeze(), negatives)
        loss = loss_pos + loss_neg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % val_steps == 0:
            train_auc = evalAUC(model, X, A, train_y, train_idx)
            val_auc = evalAUC(model, X, A, val_y, val_idx)
            if val_auc > best_auc:
                best_auc = val_auc
                stop_steps = 0
                torch.save({
                        'epoch': i,
                        'val_AUC': val_auc,
                        'model_state_dict': model.state_dict(),
                        'model_params': params
                    }, savepath)
            else:
                stop_steps += 1

        if early_stop_patience != -1 and  stop_steps >= early_stop_patience:
            break

        tqdm.set_description(iterable, desc='Loss: %.4f. Train AUC %.4f. Validation AUC: %.4f' % (loss, train_auc, val_auc))

    print(f'Best validation AUC: {best_auc}')
    if return_score:
        score = evalAUC(model, X, A, val_y, val_idx)
        return score, model
    return model

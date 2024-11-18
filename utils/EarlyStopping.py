class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, verbose = True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        elif self.best_loss - val_loss <= self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        if self.verbose:
            print("Early Stopping counter: ", self.counter)
        return self.early_stop
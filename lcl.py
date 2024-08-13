import torch
from torch import nn
import torch.nn.functional as F
from scipy.stats import special_ortho_group
from scipy.stats import chi2

# Generate data
def generate_data(n1, n2, n3, T, d, r1, r2, sigma1, sigma2, sigma3):
    # Generate Gamma
    Gamma_values = torch.rand(d) * 20 + 1
    Gamma = torch.diag(Gamma_values)
    std = torch.sqrt(Gamma)  # Calculate the standard deviation

    # Generate f
    f = torch.normal(15, 5, size=(T,))

    # Generate U0_star, U1_star, U2_star
    U0_star = torch.tensor(special_ortho_group.rvs(d), dtype=torch.float)
    U1_star = U0_star[:, :r1]
    U2_star = U0_star[:, r1:r1+r2]

    # Retrieve the chi-squared distribution quantiles for 0.7 and 0.4
    quantile_70 = chi2.ppf(0.67, r1)
    quantile_40 = chi2.ppf(0.33, r1)

    Theta1_train = torch.normal(0, sigma1, size=(r1, n1))
    S1_train = torch.normal(0, sigma2, size=(r1, n1*T))
    P_train = torch.cat([Theta1_train * fi for fi in f], dim=1) + S1_train

    Theta2_train = torch.normal(0, sigma1, size=(r2, n1))
    S2_train = torch.normal(0, sigma3, size=(r2, n1*T))
    Q_train = torch.repeat_interleave(Theta2_train, T, dim=1) + S2_train

    E_train = torch.mm(std, torch.randn(d, n1*T))  # Generate random numbers from standard normal distribution and multiply by the standard deviation

    X_train = U1_star @ P_train + U2_star @ Q_train + E_train

    norm_Theta1_train = torch.norm(Theta1_train/sigma1, dim=0, p=2) ** 2
    norm_Theta1_train += torch.randn_like(norm_Theta1_train) * 0.5  # Add noise to the norms

    # Initialize the labels tensor
    Y_train = torch.zeros(n1, dtype=torch.long)

    # Compare the 2-norms to the quantiles and assign labels
    Y_train[norm_Theta1_train >= quantile_70] = 2
    Y_train[(norm_Theta1_train < quantile_70) & (norm_Theta1_train >= quantile_40)] = 1
    # By default, the remaining labels are 0, so no extra operation is needed
    
    # Valid data
    Theta1_valid = torch.normal(0, sigma1, size=(r1, n2))
    S1_valid = torch.normal(0, sigma2, size=(r1, n2*T))
    P_valid = torch.cat([Theta1_valid * fi for fi in f], dim=1) + S1_valid

    Theta2_valid = torch.normal(0, sigma1, size=(r2, n2))
    S2_valid = torch.normal(0, sigma3, size=(r2, n2*T))
    Q_valid = torch.repeat_interleave(Theta2_valid, T, dim=1) + S2_valid

    E_valid = torch.mm(std, torch.randn(d, n2*T))  # Generate random numbers from standard normal distribution and multiply by the standard deviation

    X_valid = U1_star @ P_valid + U2_star @ Q_valid + E_valid

    norm_Theta1_valid = torch.norm(Theta1_valid/sigma1, dim=0, p=2) ** 2
    norm_Theta1_valid += torch.randn_like(norm_Theta1_valid) * 0.5  # Add noise to the norms

    # Initialize the labels tensor
    Y_valid = torch.zeros(n2, dtype=torch.long)

    # Compare the 2-norms to the quantiles and assign labels
    Y_valid[norm_Theta1_valid >= quantile_70] = 2
    Y_valid[(norm_Theta1_valid < quantile_70) & (norm_Theta1_valid >= quantile_40)] = 1

    # Test data
    Theta1_test = torch.normal(0, sigma1, size=(r1, n3))
    S1_test = torch.normal(0, sigma2, size=(r1, n3*T))
    P_test = torch.cat([Theta1_test * fi for fi in f], dim=1) + S1_test

    Theta2_test = torch.normal(0, sigma1, size=(r2, n3))
    S2_test = torch.normal(0, sigma3, size=(r2, n3*T))
    Q_test = torch.repeat_interleave(Theta2_test, T, dim=1) + S2_test

    E_test = torch.mm(std, torch.randn(d, n3*T))  # Generate random numbers from standard normal distribution and multiply by the standard deviation

    X_test = U1_star @ P_test + U2_star @ Q_test + E_test

    norm_Theta1_test = torch.norm(Theta1_test/sigma1, dim=0, p=2) ** 2
    norm_Theta1_test += torch.randn_like(norm_Theta1_test) * 0.5  # Add noise to the norms

    # Initialize the labels tensor
    Y_test = torch.zeros(n3, dtype=torch.long)

    # Compare the 2-norms to the quantiles and assign labels
    Y_test[norm_Theta1_test >= quantile_70] = 2
    Y_test[(norm_Theta1_test < quantile_70) & (norm_Theta1_test >= quantile_40)] = 1

    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test

# Generate data
def generate_data2(n1, n2, n3, T, d, r1, r2, sigma1, sigma2, sigma3):
    # Generate Gamma
    Gamma_values = torch.rand(d) * 20 + 10# Generate random numbers between 1 and 9
    Gamma = torch.diag(Gamma_values)
    std = torch.sqrt(Gamma)  # Calculate the standard deviation

    # Generate f
    f = torch.normal(15, 5, size=(T,))

    # Generate U0_star, U1_star, U2_star
    U0_star = torch.tensor(special_ortho_group.rvs(d), dtype=torch.float)
    U1_star = U0_star[:, :r1]
    U2_star = U0_star[:, r1:r1+r2]

    w1 = torch.tensor([1, 0, 0, 0, 0], dtype=torch.float)
    w2 = torch.tensor([0, 1, 0, 0, 0], dtype=torch.float)
    w3 = torch.tensor([0, 0, 0, 0, 0], dtype=torch.float)
    W = torch.stack([w1, w2, w3])

    Theta1_train = torch.normal(0, sigma1, size=(r1, n1))
    S1_train = torch.normal(0, sigma2, size=(r1, n1*T))
    P_train = torch.cat([Theta1_train * fi for fi in f], dim=1) + S1_train

    Theta2_train = torch.normal(0, sigma1, size=(r2, n1))
    S2_train = torch.normal(0, sigma3, size=(r2, n1*T))
    Q_train = torch.repeat_interleave(Theta2_train, T, dim=1) + S2_train

    E_train = torch.mm(std, torch.randn(d, n1*T))  # Generate random numbers from standard normal distribution and multiply by the standard deviation

    X_train = U1_star @ P_train + U2_star @ Q_train + E_train

    dot_products = torch.matmul(W, Theta1_train)
    probabilities = F.softmax(dot_products, dim=0)  # Apply softmax to get probabilities
    Y_train = torch.multinomial(probabilities.t(), 1).squeeze(1)
    
    # Valid data
    Theta1_valid = torch.normal(0, sigma1, size=(r1, n2))
    S1_valid = torch.normal(0, sigma2, size=(r1, n2*T))
    P_valid = torch.cat([Theta1_valid * fi for fi in f], dim=1) + S1_valid
    
    Theta2_valid = torch.normal(0, sigma1, size=(r2, n2))
    S2_valid = torch.normal(0, sigma3, size=(r2, n2*T))
    Q_valid = torch.repeat_interleave(Theta2_valid, T, dim=1) + S2_valid

    E_valid = torch.mm(std, torch.randn(d, n2*T))  # Generate random numbers from standard normal distribution and multiply by the standard deviation

    X_valid = U1_star @ P_valid + U2_star @ Q_valid + E_valid

    dot_products = torch.matmul(W, Theta1_valid)
    probabilities = F.softmax(dot_products, dim=0)  # Apply softmax to get probabilities
    Y_valid = torch.multinomial(probabilities.t(), 1).squeeze(1)

    # Test data
    Theta1_test = torch.normal(0, sigma1, size=(r1, n3))
    S1_test = torch.normal(0, sigma2, size=(r1, n3*T))
    P_test = torch.cat([Theta1_test * fi for fi in f], dim=1) + S1_test

    Theta2_test = torch.normal(0, sigma1, size=(r2, n3))
    S2_test = torch.normal(0, sigma3, size=(r2, n3*T))
    Q_test = torch.repeat_interleave(Theta2_test, T, dim=1) + S2_test

    E_test = torch.mm(std, torch.randn(d, n3*T))  # Generate random numbers from standard normal distribution and multiply by the standard deviation

    X_test = U1_star @ P_test + U2_star @ Q_test + E_test

    dot_products = torch.matmul(W, Theta1_test)
    probabilities = F.softmax(dot_products, dim=0)  # Apply softmax to get probabilities
    Y_test = torch.multinomial(probabilities.t(), 1).squeeze(1)

    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test

# Augment data
def augment_data(X, n, d):
    A = [torch.diag(torch.randint(0, 2, (d,), dtype=torch.float)) for _ in range(n)]    # Generate A
    Xi1_list = []
    Xi2_list = []
    for i in range(n):
        indices = [j for j in range(i, X.shape[1], n)]
        Xi = X[:, indices]
        Ai = A[i]
        Xi1 = Ai @ Xi
        Xi1_list.append(Xi1)
        Xi2 = Xi - Xi1
        Xi2_list.append(Xi2)
    return Xi1_list, Xi2_list

# Augment data for the matrix case
def augment_data_mat(X, n, T, d, lambda0):
    A = [torch.diag(torch.randint(0, 2, (d,), dtype=torch.float)) for _ in range(n)]    # Generate A
    Xi1_list = []
    Xi2_list = []
    mean_X1 = torch.empty((d, 0), dtype=torch.float)  # Initialize an empty tensor for concatenation
    mean_X2 = torch.empty((d, 0), dtype=torch.float)
    Z1 = torch.zeros(d, d)
    Z2 = torch.zeros(d, d)
    for i in range(n):
        indices = [j for j in range(i, X.shape[1], n)]
        Xi = X[:, indices]
        mean_Xi = Xi.mean(dim=1, keepdim=True)
        Ai = A[i]
        Xi1 = Ai @ Xi
        Xi1_list.append(Xi1)
        mean_Xi1 = Ai @ mean_Xi
        mean_X1 = torch.cat((mean_X1, mean_Xi1), dim=1)
        Xi2 = Xi - Xi1
        Xi2_list.append(Xi2)
        mean_Xi2 = mean_Xi - mean_Xi1
        mean_X2 = torch.cat((mean_X2, mean_Xi2), dim=1)
        Z1 += torch.mm(Xi1, Xi2.t()) + torch.mm(Xi2, Xi1.t())
        Z2 += Xi1 @ torch.ones(T, T) @ Xi2.t() + Xi2 @ torch.ones(T, T) @ Xi1.t()
    Z = Z1/(2*n*T*lambda0) - Z2/(2*n*T**2*lambda0)
    return Xi1_list, Xi2_list, mean_X1, mean_X2, Z

# Augment data for the matrix case
def augment_data_mat_mtimes(X, n, T, d, m, lambda0):
    Xi1_list = []
    Xi2_list = []
    mean_X1 = torch.empty((d, 0), dtype=torch.float)  # Initialize an empty tensor for concatenation
    mean_X2 = torch.empty((d, 0), dtype=torch.float)
    Z1 = torch.zeros(d, d)
    Z2 = torch.zeros(d, d)
    for _ in range(m):
        A = [torch.diag(torch.randint(0, 2, (d,), dtype=torch.float)) for _ in range(n)]    # Generate A
        for i in range(n):
            indices = [j for j in range(i, X.shape[1], n)]
            Xi = X[:, indices]
            mean_Xi = Xi.mean(dim=1, keepdim=True)
            Ai = A[i]
            Xi1 = Ai @ Xi
            Xi1_list.append(Xi1)
            mean_Xi1 = Ai @ mean_Xi
            mean_X1 = torch.cat((mean_X1, mean_Xi1), dim=1)
            Xi2 = Xi - Xi1
            Xi2_list.append(Xi2)
            mean_Xi2 = mean_Xi - mean_Xi1
            mean_X2 = torch.cat((mean_X2, mean_Xi2), dim=1)
            Z1 += torch.mm(Xi1, Xi2.t()) + torch.mm(Xi2, Xi1.t())
            Z2 += Xi1 @ torch.ones(T, T) @ Xi2.t() + Xi2 @ torch.ones(T, T) @ Xi1.t()
    Z = Z1/(2*n*T*m*lambda0) - Z2/(2*n*T**2*m*lambda0)
    return Xi1_list, Xi2_list, mean_X1, mean_X2, Z

# Mean of the data of time T
def mean_data(P, n, r):
    mean_P = torch.empty((r, 0), dtype=torch.float)  # Initialize an empty tensor for concatenation
    for i in range(n):
        indices = [j for j in range(i, P.shape[1], n)]
        Pi = P[:, indices]
        mean_Pi = Pi.mean(dim=1, keepdim=True)
        mean_P = torch.cat((mean_P, mean_Pi), dim=1)
    return mean_P

# Rearrange data
def rearrange_data(X, n):
    for i in range(n):
        indices = [j for j in range(i, X.shape[1], n)]
        Xi = X[:, indices]
        if i == 0:
            X1 = Xi
        else:
            X1 = torch.cat((X1, Xi), dim=1)
    return X1

# Interpolation of the data of time T
def interpolation_data(X, T, r):
    # Create the expanded matrices
    expanded_matrix1 = X.unsqueeze(0).expand(T, -1, -1)
    expanded_matrix2 = X.unsqueeze(1).expand(-1, T, -1)

    # Perform the subtraction which will broadcast across the T dimension
    differences = expanded_matrix1 - expanded_matrix2

    # Create an upper triangular mask excluding the diagonal
    mask = torch.triu(torch.ones(T, T), diagonal=1).to(torch.bool)

    # Use the mask to select the upper triangular elements excluding the diagonal
    # and reshape to collapse the first two dimensions
    result_matrix = differences[mask].reshape(-1, r)
    return result_matrix

# Loss function to learn the feature
def loss1_LCL(Xi1_list, Xi2_list, model, n, T):
    pos = 0
    neg = 0
    for i in range(n):
        Xi1 = Xi1_list[i].t()
        Xi2 = Xi2_list[i].t()
        pos += torch.trace(model(Xi1) @ model(Xi2).t())
        neg += torch.trace(torch.ones(T, T) @ model(Xi1) @ model(Xi2).t())
    loss = - pos / (n * T) + neg / (n * T**2)
    return loss

# Loss function to learn the feature with penalty
def loss1_LCL_pen(Xi1_list, Xi2_list, model, n, T, lambda0):
    pen = 0
    pos = 0
    neg = 0
    for i, layer in enumerate(model.model):
        if isinstance(layer, nn.Linear):
            pen += lambda0 / 2 * (torch.norm(layer.weight.t() @ layer.weight, 'fro'))**2
    for i in range(n):
        Xi1 = Xi1_list[i].t()
        Xi2 = Xi2_list[i].t()
        pos += torch.trace(model(Xi1) @ model(Xi2).t())
        neg += torch.trace(torch.ones(T, T) @ model(Xi1) @ model(Xi2).t())
    loss = pen - pos / (n * T) + neg / (n * T**2)
    return loss

# Loss function to compute W1
def loss1_LCLW1(Xi1_list, Xi2_list, model, n, T, lambda0):
    pen = lambda0 / 2 * (torch.norm(model.fc1.weight.t() @ model.fc1.weight, 'fro'))**2
    pos = 0
    neg = 0
    for i in range(n):
        Xi1 = Xi1_list[i].t()
        Xi2 = Xi2_list[i].t()
        pos += torch.trace(model(Xi1) @ model(Xi2).t())
        neg += torch.trace(torch.ones(T, T) @ model(Xi1) @ model(Xi2).t())
    loss = pen - pos / (n * T) + neg / (n * T**2)
    return loss

# Losss function to learn the feature for LSSL
def loss_LSSL(I, encoder, decoder, tau, n, r, T, lambda0):
    pen = 0
    mse_loss = F.mse_loss(I, decoder(encoder(I)), reduction='sum')
    for i in range(n):
        Ii = I[i*T:(i+1)*T, :]
        pen += F.cosine_similarity(interpolation_data(encoder(Ii), T, r), tau, dim=1).sum()
    loss = mse_loss - lambda0 * pen
    return loss

# Learn the feature for LCL with 1 layers
class LCL(nn.Module):
    def __init__(self, input_dim, hidden1_dim, output_dim):
        super(LCL, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden1_dim),
            nn.ReLU(),
            nn.Linear(hidden1_dim, output_dim)
        )

    def forward(self, x):
        x = self.model(x)
        x_normalized = F.normalize(x, p=2, dim=1)
        return x_normalized
    
class LCL2(nn.Module):
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, output_dim):
        super(LCL2, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden1_dim),
            nn.ReLU(),
            nn.Linear(hidden1_dim, hidden2_dim),
            nn.ReLU(),
            nn.Linear(hidden2_dim, output_dim)
        )

    def forward(self, x):
        x = self.model(x)
        x_normalized = F.normalize(x, p=2, dim=1)
        return x_normalized

# Learn the feature for LCL with penalty
class LCL_pen(nn.Module):
    def __init__(self, input_dim, hidden1_dim, output_dim):
        super(LCL_pen, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden1_dim),
            nn.ReLU(),
            nn.Linear(hidden1_dim, output_dim),
        )

    def forward(self, x):
        x = self.model(x)
        return x

# Learn the feature only with one layer
class LCLW1(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LCLW1, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        torch.nn.init.normal_(self.fc1.weight, mean=0, std=1)
    def forward(self, x):
        x = self.fc1(x)
        return x

# Learn the feature for LSSL
class LSSL_encoder(nn.Module):
    def __init__(self, input_dim, hidden1_dim, output_dim):
        super(LSSL_encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden1_dim),
            nn.ReLU(),
            nn.Linear(hidden1_dim, output_dim),
        )

    def forward(self, x):
        x = self.model(x)
        return x

class LSSL_decoder(nn.Module):
    def __init__(self, input_dim, hidden1_dim, output_dim):
        super(LSSL_decoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden1_dim),
            nn.ReLU(),
            nn.Linear(hidden1_dim, output_dim),
        )

    def forward(self, x):
        x = self.model(x)
        return x
    
# Learn the feature for LSSL only with one layer
class LSSL_encoder_sim(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LSSL_encoder_sim, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, x):
        x = self.model(x)
        return x

class LSSL_decoder_sim(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LSSL_decoder_sim, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, x):
        x = self.model(x)
        return x

# Classifier
class classifier(nn.Module):
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, output_dim):
        super(classifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden1_dim),
            nn.ReLU(),
            nn.Linear(hidden1_dim, hidden2_dim),
            nn.ReLU(),
            nn.Linear(hidden2_dim, output_dim)
        )
    def forward(self, x):
        return self.model(x)
    
# Classifier
class classifier2(nn.Module):
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, hidden3_dim,output_dim):
        super(classifier2, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden1_dim),
            nn.ReLU(),
            nn.Linear(hidden1_dim, hidden2_dim),
            nn.ReLU(),
            nn.Linear(hidden2_dim, hidden3_dim),
            nn.ReLU(),
            nn.Linear(hidden3_dim, output_dim)
        )
    def forward(self, x):
        return self.model(x)
    
class classifier3(nn.Module):
    def __init__(self, input_dim, hidden1_dim, output_dim):
        super(classifier3, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden1_dim),
            nn.ReLU(),
            nn.Linear(hidden1_dim, output_dim)
        )
    def forward(self, x):
        return self.model(x)

# Train the encoder of LCL
def Train_encoder_LCL(model_encoder, loss1, Xi1_list, Xi2_list, n, T, lambda0, epochs):
    optimizer = torch.optim.Adam(model_encoder.parameters(), lr=1)
    # Perform optimization
    for step in range(epochs):  # For example, iterate 1000 times
        optimizer.zero_grad()  # Clear the gradients from the previous step
        loss = loss1(Xi1_list, Xi2_list, model_encoder, n, T, lambda0)  # Compute the loss
        loss.backward()  # Compute gradients
        optimizer.step()  # Update parameters
        # Optionally, print loss every 100 steps
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item()}")
    return model_encoder

# Train the encoder and decoder of LSSL
def Train_coder_LSSL(model_encoder, model_decoder, loss1, X, n, r, T, lambda0, epochs):
    tau = nn.Parameter(torch.rand(1))
    params = list(model_encoder.parameters()) + list(model_decoder.parameters()) + [tau]
    optimizer = torch.optim.Adam(params, lr=0.1)
    # Perform optimization
    for step in range(epochs):
        optimizer.zero_grad()
        loss = loss1(X, model_encoder, model_decoder, tau, n, r, T, lambda0)
        loss.backward()
        optimizer.step()
        # print loss every 100 steps
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item()}")
    return model_encoder, model_decoder

# Train the classifier
def Train_classifier(model_encoder, model_classifier, mean_P_fun, X, Y, n, X_val, Y_val, n_val, r, epochs, epochs_max, desired_accuracy, desired_accuracy_val):
    P = model_encoder(X.t()).detach()
    mean_P = mean_P_fun(P.t(), n, r).t()

    # Validation
    P_val = model_encoder(X_val.t()).detach()
    mean_P_val = mean_P_fun(P_val.t(), n_val, r).t()
    optimizer = torch.optim.Adam(model_classifier.parameters(), lr=0.01)
    loss1 = nn.CrossEntropyLoss()
    # Perform optimization
    for step in range(epochs):  # For example, iterate 1000 times
        optimizer.zero_grad()  # Clear the gradients from the previous step
        loss = loss1(model_classifier(mean_P), Y)  # Compute the loss
        loss.backward()  # Compute gradients
        optimizer.step()  # Update parameters
        # Optionally, print loss every 100 steps
        if step % 100 == 0:
            with torch.no_grad():
                print(f"Step {step}, Loss: {loss.item()}")
                _, predicted = torch.max(model_classifier(mean_P), 1)
                correct = (predicted == Y).sum().item()
                total = Y.size(0)
                accuracy = correct / total
                print(f'Accuracy: {accuracy * 100:.2f}%')
                _, predicted_val = torch.max(model_classifier(mean_P_val), 1)
                correct_val = (predicted_val == Y_val).sum().item()
                accuracy_val = correct_val / Y_val.size(0)
                print(f'Accuracy_val: {accuracy_val * 100:.2f}%')
    if accuracy < desired_accuracy or accuracy_val < desired_accuracy_val:
        for step in range(epochs, epochs_max):
            optimizer.zero_grad()  # Clear the gradients from the previous step
            loss = loss1(model_classifier(mean_P), Y)  # Compute the loss
            loss.backward()  # Compute gradients
            optimizer.step()  # Update parameters
            # Optionally, print loss every 100 steps
            if step % 100 == 0:
                with torch.no_grad():
                    print(f"Step {step}, Loss: {loss.item()}")
                    _, predicted = torch.max(model_classifier(mean_P), 1)
                    correct = (predicted == Y).sum().item()
                    total = Y.size(0)
                    accuracy = correct / total
                    print(f'Accuracy: {accuracy * 100:.2f}%')
                    _, predicted_val = torch.max(model_classifier(mean_P_val), 1)
                    correct_val = (predicted_val == Y_val).sum().item()
                    accuracy_val = correct_val / Y_val.size(0)
                    print(f'Accuracy_val: {accuracy_val * 100:.2f}%')

                if accuracy >= desired_accuracy and accuracy_val >= desired_accuracy_val:
                    print("Reached desired accuracy, stopping training.")
                    break
    return model_classifier

# Compute the accuracy
def accuracy(model_encoder, model_classifier, mean_P_fun, X, Y, n, r):
    P = model_encoder(X.t()).detach()
    mean_P = mean_P_fun(P.t(), n, r).t()
    _, predicted = torch.max(model_classifier(mean_P), 1)
    correct = (predicted == Y).sum().item()
    total = Y.size(0)
    accuracy = correct / total
    print(accuracy)
    return accuracy

def mean_per_class_accuracy(model_encoder, model_classifier, mean_p_fun, X, Y, n, r):
    P = model_encoder(X.t()).detach()
    mean_P = mean_p_fun(P.t(), n, r).t()
    outputs = model_classifier(mean_P)
    _, predicted = torch.max(outputs, 1)
    
    C = torch.max(Y) + 1  # Assuming class labels start from 0
    accuracy_per_class = torch.zeros(C)
    
    for c in range(C):
        idx = (Y == c)
        true_positives = torch.sum(predicted[idx] == Y[idx]).item()
        condition_positive = torch.sum(idx).item()
        accuracy_per_class[c] = true_positives / condition_positive
    
    mean_accuracy = torch.mean(accuracy_per_class)
    print(f'Mean Per-Class Accuracy: {mean_accuracy}')
    return mean_accuracy
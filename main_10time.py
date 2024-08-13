import torch
from torch import nn
import lcl
import lssl
for i in range(15):
    print(i)
    n1 = 2000
    n2 = 200
    n3 = 200
    T = 5
    d = 100
    r1 = 5
    r2 = 8
    sigma1 = 2
    sigma2 = 2
    sigma3 = 0.002
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = lcl.generate_data(n1, n2, n3, T, d, r1, r2, sigma1, sigma2, sigma3)
    m = 1
    lambda0 = 1
    lambda1 = 5

    # Define dimensions
    input_dim_LCL = d
    hidden1_dim_LCL = 80 
    hidden2_dim_LCL = 50 
    output_dim_LCL = r1
    classes = 3

    # Augment data
    Xi1_list, Xi2_list, mean_X1, mean_X2, Z= lcl.augment_data_mat_mtimes(X_train, n1, T, d, m, lambda0)
    eigen21 = torch.linalg.eigvalsh(Z) ** 2
    print(-torch.sum(eigen21[-r1:])*lambda0/2)

    # Initialize the model
    model_encoder_LCLW1 = lcl.LCL2(input_dim_LCL, hidden1_dim_LCL, hidden2_dim_LCL, output_dim_LCL)
    optimizer1 = torch.optim.Adam(model_encoder_LCLW1.parameters(), lr=1)

    # Perform optimization
    for step in range(500):
        optimizer1.zero_grad()  # Clear the gradients from the previous step
        loss = lcl.loss1_LCL(Xi1_list, Xi2_list, model_encoder_LCLW1, n1*m, T)  # Compute the loss
        loss.backward()  # Compute gradients
        optimizer1.step()  # Update parameters
    
        # Optionally, print loss every 100 steps
        if step % 100 == 0:
            with open('output.txt', 'a') as file:
                print(f"Step {step}, Loss: {loss.item()}", file=file)
    P = model_encoder_LCLW1(X_train.t()).detach()
    mean_P = lcl.mean_data(P.t(), n1, output_dim_LCL).t()
    model_classifier_LCLW1 = lcl.classifier2(output_dim_LCL, 50, 50, 50, classes)
    optimizer2 = torch.optim.Adam(model_classifier_LCLW1.parameters(), lr=0.01)
    loss2 = nn.CrossEntropyLoss()
    # Perform optimization
    for step in range(3000):
        optimizer2.zero_grad()  # Clear the gradients from the previous step
        loss = loss2(model_classifier_LCLW1(mean_P), Y_train)  # Compute the loss
        loss.backward()  # Compute gradients
        optimizer2.step()  # Update parameters
        # Optionally, print loss every 100 steps
        if step % 100 == 0:
            with open('output.txt', 'a') as file:
                print(f"Step {step}, Loss: {loss.item()}", file=file)
            _, predicted = torch.max(model_classifier_LCLW1(mean_P), 1)
            correct = (predicted == Y_train).sum().item()
            total = Y_train.size(0)
            accuracy = correct / total
            with open('output.txt', 'a') as file:
                print(f'Accuracy: {accuracy * 100:.2f}%', file=file)
            _, predicted = torch.max(model_classifier_LCLW1(lcl.mean_data(model_encoder_LCLW1(X_valid.t()).detach().t(), n2, output_dim_LCL).t()), 1)
            correct = (predicted == Y_valid).sum().item()
            total = Y_valid.size(0)
            accuracy = correct / total
            with open('output.txt', 'a') as file:
                print(f'Accuracy: {accuracy * 100:.2f}%', file=file)
            _, predicted = torch.max(model_classifier_LCLW1(lcl.mean_data(model_encoder_LCLW1(X_test.t()).detach().t(), n3, output_dim_LCL).t()), 1)
            correct = (predicted == Y_test).sum().item()
            total = Y_test.size(0)
            accuracy = correct / total
            with open('output.txt', 'a') as file:
                print(f'Accuracy: {accuracy * 100:.2f}%', file=file)

    model_classifier_LCLW13 = lcl.classifier3(output_dim_LCL, 50, classes)
    optimizer6 = torch.optim.Adam(model_classifier_LCLW13.parameters(), lr=0.01)
    loss2 = nn.CrossEntropyLoss()
    # Perform optimization
    for step in range(3000):
        optimizer6.zero_grad()  # Clear the gradients from the previous step
        loss = loss2(model_classifier_LCLW13(mean_P), Y_train)  # Compute the loss
        loss.backward()  # Compute gradients
        optimizer6.step()  # Update parameters
        # Optionally, print loss every 100 steps
        if step % 100 == 0:
            with open('output.txt', 'a') as file:
                print(f"Step {step}, Loss: {loss.item()}", file=file)
            _, predicted = torch.max(model_classifier_LCLW13(mean_P), 1)
            correct = (predicted == Y_train).sum().item()
            total = Y_train.size(0)
            accuracy = correct / total
            with open('output.txt', 'a') as file:
                print(f'Accuracy: {accuracy * 100:.2f}%', file=file)
            _, predicted = torch.max(model_classifier_LCLW13(lcl.mean_data(model_encoder_LCLW1(X_valid.t()).detach().t(), n2, output_dim_LCL).t()), 1)
            correct = (predicted == Y_valid).sum().item()
            total = Y_valid.size(0)
            accuracy = correct / total
            with open('output.txt', 'a') as file:
                print(f'Accuracy: {accuracy * 100:.2f}%', file=file)
            _, predicted = torch.max(model_classifier_LCLW13(lcl.mean_data(model_encoder_LCLW1(X_test.t()).detach().t(), n3, output_dim_LCL).t()), 1)
            correct = (predicted == Y_test).sum().item()
            total = Y_test.size(0)
            accuracy = correct / total
            with open('output.txt', 'a') as file:
                print(f'Accuracy: {accuracy * 100:.2f}%', file=file)
    input_dim_LSSL = d
    hidden1_dim_LSSL = 100
    hidden2_dim_LSSL = 80
    hidden3_dim_LSSL = 50
    output_dim_LSSL = r1 + r2

    # Initialize the model
    tau = nn.Parameter(torch.rand(1))
    X_train_re = lssl.rearrange_data(X_train, n1).t()
    model_encoder_LSSL = lssl.LSSL_encoder(input_dim_LSSL, hidden3_dim_LSSL, output_dim_LSSL)
    model_decoder_LSSL = lssl.LSSL_decoder(output_dim_LSSL, hidden3_dim_LSSL, input_dim_LSSL)
    #model_encoder_LSSL = lssl.LSSL_encoder_sim(input_dim_LSSL, output_dim_LSSL)
    #model_decoder_LSSL = lssl.LSSL_decoder_sim(output_dim_LSSL, input_dim_LSSL)
    params = list(model_encoder_LSSL.parameters()) + list(model_decoder_LSSL.parameters()) + [tau]
    optimizer3 = torch.optim.Adam(params, lr=0.1)

    # Perform optimization
    for step in range(500):
        optimizer3.zero_grad()  # Clear the gradients from the previous step
        loss = lssl.loss_LSSL(X_train_re, model_encoder_LSSL, model_decoder_LSSL, tau, n1, output_dim_LSSL, T, lambda1)  # Compute the loss
        loss.backward()  # Compute gradients
        optimizer3.step()  # Update parameters
        # Optionally, print loss every 100 steps
        if step % 100 == 0:
            with open('output.txt', 'a') as file:
                print(f"Step {step}, Loss: {loss.item()}", file=file)

    P = model_encoder_LSSL(X_train.t()).detach()
    mean_P = lcl.mean_data(P.t(), n1, output_dim_LSSL).t()
    model_classifier_LSSL = lcl.classifier2(output_dim_LSSL, 50, 50, 50,classes)
    optimizer4 = torch.optim.Adam(model_classifier_LSSL.parameters(), lr=0.01)
    loss2 = nn.CrossEntropyLoss()
    # Perform optimization
    for step in range(1000):
        optimizer4.zero_grad()  # Clear the gradients from the previous step
        loss = loss2(model_classifier_LSSL(mean_P), Y_train)  # Compute the loss
        loss.backward()  # Compute gradients
        optimizer4.step()  # Update parameters
        # Optionally, print loss every 100 steps
        if step % 100 == 0:
            with open('output.txt', 'a') as file:
                print(f"Step {step}, Loss: {loss.item()}", file=file)
            _, predicted = torch.max(model_classifier_LSSL(mean_P), 1)
            correct = (predicted == Y_train).sum().item()
            total = Y_train.size(0)
            accuracy = correct / total
            with open('output.txt', 'a') as file:
                print(f'Accuracy: {accuracy * 100:.2f}%', file=file)
            _, predicted = torch.max(model_classifier_LSSL(lcl.mean_data(model_encoder_LSSL(X_valid.t()).detach().t(), n2, output_dim_LSSL).t()), 1)
            correct = (predicted == Y_valid).sum().item()
            total = Y_valid.size(0)
            accuracy = correct / total
            with open('output.txt', 'a') as file:
                print(f'Accuracy: {accuracy * 100:.2f}%', file=file)
            _, predicted = torch.max(model_classifier_LSSL(lcl.mean_data(model_encoder_LSSL(X_test.t()).detach().t(), n3, output_dim_LSSL).t()), 1)
            correct = (predicted == Y_test).sum().item()
            total = Y_test.size(0)
            accuracy = correct / total
            with open('output.txt', 'a') as file:
                print(f'Accuracy: {accuracy * 100:.2f}%', file=file)

    model_classifier_LSSL2 = lcl.classifier3(output_dim_LSSL, 50, classes)
    optimizer7 = torch.optim.Adam(model_classifier_LSSL2.parameters(), lr=0.01)
    loss2 = nn.CrossEntropyLoss()
    # Perform optimization
    for step in range(1000):
        optimizer7.zero_grad()  # Clear the gradients from the previous step
        loss = loss2(model_classifier_LSSL2(mean_P), Y_train)  # Compute the loss
        loss.backward()  # Compute gradients
        optimizer7.step()  # Update parameters
        # Optionally, print loss every 100 steps
        if step % 100 == 0:
            with open('output.txt', 'a') as file:
                print(f"Step {step}, Loss: {loss.item()}", file=file)
            _, predicted = torch.max(model_classifier_LSSL2(mean_P), 1)
            correct = (predicted == Y_train).sum().item()
            total = Y_train.size(0)
            accuracy = correct / total
            with open('output.txt', 'a') as file:
                print(f'Accuracy: {accuracy * 100:.2f}%', file=file)
            _, predicted = torch.max(model_classifier_LSSL2(lcl.mean_data(model_encoder_LSSL(X_valid.t()).detach().t(), n2, output_dim_LSSL).t()), 1)
            correct = (predicted == Y_valid).sum().item()
            total = Y_valid.size(0)
            accuracy = correct / total
            with open('output.txt', 'a') as file:
                print(f'Accuracy: {accuracy * 100:.2f}%', file=file)
            _, predicted = torch.max(model_classifier_LSSL2(lcl.mean_data(model_encoder_LSSL(X_test.t()).detach().t(), n3, output_dim_LSSL).t()), 1)
            correct = (predicted == Y_test).sum().item()
            total = Y_test.size(0)
            accuracy = correct / total
            with open('output.txt', 'a') as file:
                print(f'Accuracy: {accuracy * 100:.2f}%', file=file)
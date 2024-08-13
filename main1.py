import torch
from torch import nn
import lcl
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    n1 = 2000
    n2 = 200
    n3 = 200
    m = 1 # augmentation times
    T = 5
    d = 100
    r1 = 5
    r2 = 8
    sigma1 = 2
    sigma2 = 2
    sigma3 = 0.002
    # Penalization parameter
    lambda0 = 1
    lambda1 = 5
    input_dim_LCL = d
    hidden_dim_LCL = 50 
    output_dim_LCL = r1
    input_dim_LSSL = d
    hidden1_dim_LSSL = 100
    hidden2_dim_LSSL = 80
    hidden3_dim_LSSL = 50
    output_dim_LSSL = r1 + r2
    classes = 3
    times = 30
    epochs_feature = 200
    epochs_classifier = 800
    epochs_classifier_LSSL = 500
    epochs_max = 2000
    epochs_max_LSSL = 1000
    desired_accuracy = 0.92
    desired_accuracy_LSSL = 0.92
    desired_accuracy_val = 0.8

    accuracies_LCL = torch.zeros(times)
    accuracies_LSSL = torch.zeros(times)
    for i in range(times):
        print(i+1)
        X_train, Y_train, X_valid, Y_valid, X_test, Y_test = lcl.generate_data(n1, n2, n3, T, d, r1, r2, sigma1, sigma2, sigma3)

        # Augment data
        Xi1_list, Xi2_list, mean_X1, mean_X2, Z= lcl.augment_data_mat_mtimes(X_train, n1, T, d, m, lambda0)
        eigen21 = torch.linalg.eigvalsh(Z) ** 2
        print(-torch.sum(eigen21[-r1:])*lambda0/2)

        #Train the encoder
        model_encoder_LCLW1 = lcl.LCLW1(input_dim_LCL, output_dim_LCL)
        model_encoder_LCLW1 = lcl.Train_encoder_LCL(model_encoder_LCLW1, lcl.loss1_LCLW1, Xi1_list, Xi2_list, n1*m, T, lambda0, epochs_feature)

        #Train the classifier
        model_classifier_LCLW1 = lcl.classifier2(output_dim_LCL, 50, 50, 50, classes)

        model_classifier_LCLW1 = lcl.Train_classifier(model_encoder_LCLW1, model_classifier_LCLW1, lcl.mean_data, X_train, Y_train, n1, X_valid, Y_valid, n2, r1, epochs_classifier, epochs_max, desired_accuracy, desired_accuracy_val)
        
        # Test the classifier
        accuracies_LCL[i] = lcl.accuracy(model_encoder_LCLW1, model_classifier_LCLW1, lcl.mean_data, X_test, Y_test, n3, output_dim_LCL)

        # Initialize the model
        X_train_re = lcl.rearrange_data(X_train, n1).t()
        model_encoder_LSSL = lcl.LSSL_encoder_sim(input_dim_LSSL, output_dim_LSSL)
        model_decoder_LSSL = lcl.LSSL_decoder_sim(output_dim_LSSL, input_dim_LSSL)

        # Train the model of the LSSL
        model_encoder_LSSL, model_decoder_LSSL = lcl.Train_coder_LSSL(model_encoder_LSSL, model_decoder_LSSL, lcl.loss_LSSL, X_train_re, n1, output_dim_LSSL, T, lambda1, epochs_feature)
        
        # Train the classifier
        model_classifier_LSSL = lcl.classifier2(output_dim_LSSL, 50, 50, 50, classes)
        model_classifier_LSSL = lcl.Train_classifier(model_encoder_LSSL, model_classifier_LSSL, lcl.mean_data, X_train, Y_train, n1, X_valid, Y_valid, n2,output_dim_LSSL, epochs_classifier_LSSL, epochs_max_LSSL, desired_accuracy_LSSL, desired_accuracy_val)
        
        # Test the classifier
        accuracies_LSSL[i] = lcl.accuracy(model_encoder_LSSL, model_classifier_LSSL, lcl.mean_data, X_test, Y_test, n3, output_dim_LSSL)
    mean_accuracy_LCL = torch.mean(accuracies_LCL)
    std_accuracy_LCL = torch.std(accuracies_LCL)
    print(accuracies_LCL)
    print(f"Mean accuracy LCL: {mean_accuracy_LCL}, std accuracy LCL: {std_accuracy_LCL}")
    mean_accuracy_LSSL = torch.mean(accuracies_LSSL)
    std_accuracy_LSSL = torch.std(accuracies_LSSL)
    print(accuracies_LSSL)
    print(f"Mean accuracy LSSL: {mean_accuracy_LSSL}, std accuracy LSSL: {std_accuracy_LSSL}")

    accuracies_LCL_np = accuracies_LCL.numpy()
    accuracies_LSSL_np = accuracies_LSSL.numpy()

    # 使用Matplotlib绘制箱形图
    plt.boxplot([accuracies_LCL_np, accuracies_LSSL_np], labels=['LCL', 'LSSL'])
    plt.title('Comparison of Accuracy Between LCL and LSSL')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()
import numpy as np
import warnings
from matplotlib import pylab
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import matplotlib
import cv2 as cv

warnings.filterwarnings("ignore")

no_of_dataset = 2


def stats(val):
    v = np.zeros(5)
    v[0] = max(val)
    v[1] = min(val)
    v[2] = np.mean(val)
    v[3] = np.median(val)
    v[4] = np.std(val)
    return v


def plot_conv():
    matplotlib.use('Qt5Agg')
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'CO-HC-AEB7-SA', 'SCO-HC-AEB7-SA', 'GOA-HC-AEB7-SA', 'ZOA-HC-AEB7-SA',
                 'PFZO-HC-AEB7-SA']
    Terms = ['Worst', 'Best', 'Mean', 'Median', 'Std']
    for n in range(Fitness.shape[0]):
        Conv_Graph = np.zeros((Fitness.shape[1], 5))
        for j in range(len(Algorithm) - 1):
            Conv_Graph[j, :] = stats(Fitness[n, j, :])

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('-------------------------------------------------- Statistical Report Dataset ' + str(n + 1),
              ' --------------------------------------------------')
        print(Table)
    for n in range(Fitness.shape[0]):
        length = np.arange(Fitness.shape[-1])
        Conv_Graph = Fitness[n]
        plt.plot(length, Conv_Graph[0, :], color='#f97306', linewidth=3, label='CO-HC-AEB7-SA')
        plt.plot(length, Conv_Graph[1, :], color='#3d7afd', linewidth=3, label='SCO-HC-AEB7-SA')
        plt.plot(length, Conv_Graph[2, :], color='#b9ff66', linewidth=3, label='GOA-HC-AEB7-SA')
        plt.plot(length, Conv_Graph[3, :], color='#bb3f3f', linewidth=3, label='ZOA-HC-AEB7-SA')
        plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, label='PFZO-HC-AEB7-SA')
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function')
        plt.legend(loc=1)
        plt.savefig("./Results/Convergence_Dataset_%s.png" % (n + 1))
        fig = pylab.gcf()
        fig.canvas.manager.set_window_title('Convergence curve')
        plt.show()


def Plot_Kfold_ERROR():
    eval = np.load('Eval_All_Fold_error.npy', allow_pickle=True)
    Terms = ['MEP', 'SMAPE', 'RMSE', 'MASE', 'MAE', 'MSE', 'NMSE', 'ONENORM', 'TWONORM', 'INFINITYNORM', 'MAPE',
             'Accuracy']
    Graph_Term = [0, 2, 3, 4, 5, 6, 11]
    for n in range(eval.shape[0]):
        for j in range(len(Graph_Term)):
            Graph = np.zeros((eval.shape[1], eval.shape[2]))
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    Graph[k, l] = eval[n, k, l, Graph_Term[j]]

            length = np.arange(Graph.shape[0])
            fig = plt.figure()
            ax = fig.add_axes([0.15, 0.1, 0.7, 0.8])

            ax.plot(length, Graph[:, 0], color='#ceb301', linewidth=4, marker='*', markerfacecolor='blue',
                    markersize=12,
                    label="CO-HC-AEB7-SA")
            ax.plot(length, Graph[:, 1], color='#04d8b2', linewidth=4, marker='*', markerfacecolor='red', markersize=12,
                    label="SCO-HC-AEB7-SA")
            ax.plot(length, Graph[:, 2], color='#a2cffe', linewidth=4, marker='*', markerfacecolor='#01386a',
                    markersize=12,
                    label="GOA-HC-AEB7-SA")
            ax.plot(length, Graph[:, 3], color='#b9a281', linewidth=4, marker='*', markerfacecolor='yellow',
                    markersize=12,
                    label="ZOA-HC-AEB7-SA")
            ax.plot(length, Graph[:, 4], color='black', linewidth=4, marker='*', markerfacecolor='cyan', markersize=12,
                    label="PFZO-HC-AEB7-SA")

            plt.xticks(length, ('1', '2', '3', '4', '5'))
            plt.xlabel('K fold', fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
            plt.ylabel(Terms[Graph_Term[j]], fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=3, fancybox=True, shadow=False)
            path = "./Results/K fold_Dataset_%s_%s_line_error.png" % (n + 1, Terms[Graph_Term[j]])
            plt.savefig(path)
            fig = pylab.gcf()
            fig.canvas.manager.set_window_title('K fold vs ' + Terms[Graph_Term[j]])
            plt.show()

            fig = plt.figure()
            ax = fig.add_axes([0.15, 0.1, 0.7, 0.8])
            X = np.arange(Graph.shape[0])

            ax.bar(X + 0.00, Graph[:, 5], color='#d648d7', edgecolor='w', width=0.15, label="VGG16")
            ax.bar(X + 0.15, Graph[:, 6], color='#82cafc', edgecolor='w', width=0.15, label="ANN")
            ax.bar(X + 0.30, Graph[:, 7], color='#12e193', edgecolor='w', width=0.15, label="CNN")
            ax.bar(X + 0.45, Graph[:, 8], color='#2c6fbb', edgecolor='w', width=0.15, label="HC-EB7-SA")
            ax.bar(X + 0.60, Graph[:, 4], color='k', edgecolor='w', width=0.15, label="PFZO-HC-AEB7-SA")

            plt.xticks(X + 0.15, ('1', '2', '3', '4', '5'))
            plt.xlabel('K fold', fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
            plt.ylabel(Terms[Graph_Term[j]], fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=3, fancybox=True, shadow=False)
            path = "./Results/K fold_Dataset_%s_%s_bar_error.png" % (n + 1, Terms[Graph_Term[j]])
            plt.savefig(path)
            fig = pylab.gcf()
            fig.canvas.manager.set_window_title('K fold vs ' + Terms[Graph_Term[j]])
            plt.show()


def Plot_batchsize_error():
    eval = np.load('Eval_All_BS_error.npy', allow_pickle=True)

    Terms = ['MEP', 'SMAPE', 'RMSE', 'MASE', 'MAE', 'MSE', 'NMSE', 'ONENORM', 'TWONORM', 'INFINITYNORM', 'MAPE',
             'Accuracy']
    Table_Term = [0, 1, 2, 10, 11]
    Batch_size = ['4', '8', '16', '32', '48', '64']
    Algorithm = ['TERMS', 'CO-HC-AEB7-SA', 'SCO-HC-AEB7-SA', 'GOA-HC-AEB7-SA', 'ZOA-HC-AEB7-SA', 'PFZO-HC-AEB7-SA']
    Classifier = ['TERMS', 'VGG16', 'ANN', 'CNN', 'HC-EB7-SA', 'PFZO-HC-AEB7-SA']
    for n in range(eval.shape[0]):
        for k in range(eval.shape[1]):
            value = eval[n, k, :, :]
            Table = PrettyTable()
            Table.add_column(Algorithm[0], (np.asarray(Terms))[np.asarray(Table_Term)])
            for j in range(len(Algorithm) - 1):
                Table.add_column(Algorithm[j + 1], value[j, Table_Term])
            print('-------------------------------------------------- ', str(Batch_size[k]), ' Batch size ',
                  'Algorithm Comparison of Dataset ', str(n + 1), '--------------------------------------------------')
            print(Table)
            Table = PrettyTable()
            Table.add_column(Classifier[0], (np.asarray(Terms))[np.asarray(Table_Term)])
            for j in range(len(Classifier) - 1):
                Table.add_column(Classifier[j + 1], value[len(Algorithm) + j - 1, Table_Term])
            print('-------------------------------------------------- ', str(Batch_size[k]), ' Batch size ',
                  'Classifier Comparison of Dataset ', str(n + 1), '--------------------------------------------------')
            print(Table)


def Image_comparision():
    for n in range(no_of_dataset):
        Original = np.load('Images_' + str(n + 1) + '.npy', allow_pickle=True)
        Images_1 = [79, 80, 89, 111, 124]
        Images_2 = [97, 102, 123, 152, 159]
        Images = [Images_1, Images_2]
        Images = np.asarray(Images)
        for i in range(len(Images[n])):
            print(i, len(Images))
            Image = Original[Images[n][i]]
            Orig = Image.copy()
            Img = Image.copy()
            # Converting image to grayscale
            gray = cv.cvtColor(Img, cv.COLOR_BGR2GRAY)
            sift = cv.SIFT_create()
            kp = sift.detect(gray, None)
            # Marking the keypoint on the image using circles
            sift_img = cv.drawKeypoints(gray, kp, Img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            plt.subplot(1, 2, 1).axis('off')
            plt.imshow(Orig)
            plt.title('Original Image', fontsize=10)

            plt.subplot(1, 2, 2).axis('off')
            plt.imshow(sift_img)
            plt.title('SIFT Images', fontsize=10)

            path = "./Results/Image_Results/Dataset_%s_image_%s.png" % (n + 1, i + 1)
            plt.savefig(path)
            plt.show()

            cv.imwrite('./Results/Image_Results/Dataset_' + str(n + 1) + '_Original_image_' + str(i + 1) + '.png',
                       Orig)
            cv.imwrite('./Results/Image_Results/Dataset_' + str(n + 1) + '_SIFT_Image_' + str(i + 1) + '.png',
                       sift_img)


if __name__ == '__main__':
    plot_conv()
    Plot_Kfold_ERROR()
    Plot_batchsize_error()
    Image_comparision()

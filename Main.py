import numpy as np
import os
import cv2 as cv
from numpy import matlib
from CO import CO
from GOA import GaOA
from Global_Vars import Global_Vars
from Model_ANN import Model_ANN
from Model_CNN import Model_CNN
from Model_HC_AEB7_SAT import Model_HC_AEB7_SAT
from Model_VGG16 import Model_VGG16
from Obj_fun import objfun_cls
from PROPOSED import PROPOSED
from Plot_results import *
from SCO import SCO
from ZOA import ZOA
from whole_face import extract_whole_face

no_of_dataset = 2

# Read the Dataset
an = 0
if an == 1:
    Dataset_fold = './Datasets/Dataset_1/faces_02/part3/'
    img_dir = os.listdir(Dataset_fold)
    Images = []
    Target = []
    for n in range(len(img_dir)):
        print(n, len(img_dir))
        Image_path = Dataset_fold + img_dir[n]
        img = cv.imread(Image_path)
        img = cv.resize(img, (512, 512))
        img = np.uint8(img)
        name = (Image_path.split('/')[-1]).split('_')[-4]
        tar = name
        Images.append(img)
        Target.append(tar)
    Images = np.asarray(Images)
    Target = np.asarray(Target)
    Target = (Target.astype('int')).reshape(-1, 1)
    np.save('Images_1.npy', Images)
    np.save('Targets_1.npy', Target)

# Read the Dataset_2
an = 0
if an == 1:
    Dataset_fold = './Datasets/Dataset_2/face_age/'
    Image_fold = os.listdir(Dataset_fold)
    Images = []
    Target = []
    for n in range(len(Image_fold)):
        img_files = Dataset_fold + Image_fold[n]
        img_path = os.listdir(img_files)
        for j in range(len(img_path)):
            print(n, len(Image_fold), j, len(img_path))
            img_dir = img_files + '/' + img_path[j]
            img = cv.imread(img_dir)
            img = cv.resize(img, (512, 512))
            img = np.uint8(img)
            name = img_dir.split('/')[-2]
            Images.append(img)
            Target.append(name)
    Images = np.asarray(Images)
    Target = np.asarray(Target)
    Target = (Target.astype('int')).reshape(-1, 1)
    np.save('Images_2.npy', Images)
    np.save('Targets_2.npy', Target)


# For Face Images
an = 0
if an == 1:
    for n in range(no_of_dataset):
        Image = np.load('Images_' + str(n + 1) + '.npy', allow_pickle=True)
        SIFT_Images = []
        len_kp = []
        for i in range(len(Image)):  # len(Image)
            print(i, len(Image))
            image = Image[i]
            # Detect Facial points
            whole_face = extract_whole_face(image)
            whole_face = cv.resize(whole_face, [image.shape[0], image.shape[1]])
            Facila_points = whole_face
            # Extract SIFT
            gray = cv.cvtColor(Facila_points, cv.COLOR_BGR2GRAY)
            sift = cv.SIFT_create()
            kp = sift.detect(gray, None)
            SIFT_Images.append(kp)
            len_kp.append(len(kp))

        SIFT_Feature = []
        max_sift = np.max(len_kp)
        for j in range(len(SIFT_Images)):
            feat = SIFT_Images[j]
            Feature = np.zeros((max_sift, 7))
            for k in range(len(feat)):
                Feature[k, :] = [feat[k].angle, feat[k].class_id, feat[k].octave, feat[k].pt[0], feat[k].pt[1],
                                 feat[k].response, feat[k].size]
            SIFT_Feature.append(Feature)
        np.save('Feature_' + str(n + 1) + '.npy', SIFT_Feature)

# optimization For classification
an = 0
if an == 1:
    FITNESS = []
    BESTSOL = []
    for n in range(no_of_dataset):
        Feat_1 = np.load('Feature_' + str(n + 1) + '.npy', allow_pickle=True)
        Feat_2 = np.load('Images_' + str(n + 1) + '.npy', allow_pickle=True)
        Target = np.load('Targets_' + str(n + 1) + '.npy', allow_pickle=True)
        Global_Vars.Feat_1 = Feat_1
        Global_Vars.Feat_2 = Feat_2
        Global_Vars.Target = Target
        Npop = 10
        Chlen = 3  # Hidden Neuron Count, Epoch, Steps per epoch in HC-AEB7-SAT
        xmin = matlib.repmat(np.asarray([5, 5, 50]), Npop, 1)
        xmax = matlib.repmat(np.asarray([255, 50, 250]), Npop, 1)
        fname = objfun_cls
        initsol = np.zeros((Npop, Chlen))
        for p1 in range(initsol.shape[0]):
            for p2 in range(initsol.shape[1]):
                initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
        Max_iter = 50

        print("CO...")
        [bestfit1, fitness1, bestsol1, time1] = CO(initsol, fname, xmin, xmax, Max_iter)  # CO

        print("SCO...")
        [bestfit2, fitness2, bestsol2, time2] = SCO(initsol, fname, xmin, xmax, Max_iter)  # SCO

        print("GaOA...")
        [bestfit3, fitness3, bestsol3, time3] = GaOA(initsol, fname, xmin, xmax, Max_iter)  # GaOA

        print("ZOA...")
        [bestfit4, fitness4, bestsol4, time4] = ZOA(initsol, fname, xmin, xmax, Max_iter)  # ZOA

        print("PROPOSED...")
        [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)  # Improved ZOA

        BestSol_CLS = [bestsol1.squeeze(), bestsol2.squeeze(), bestsol3.squeeze(), bestsol4.squeeze(),
                       bestsol5.squeeze()]
        fitness = [fitness1.squeeze(), fitness2.squeeze(), fitness3.squeeze(), fitness4.squeeze(), fitness5.squeeze()]

        BESTSOL.append(BestSol_CLS)
        FITNESS.append(fitness)

    np.save('Fitness.npy', np.asarray(FITNESS))
    np.save('BestSol_CLS.npy', np.asarray(BESTSOL))


# KFOLD - Prediction
an = 0
if an == 1:
    EVAL_ALL = []
    for n in range(no_of_dataset):
        Feat_1 = np.load('Feature_' + str(n + 1) + '.npy', allow_pickle=True)[:250]
        Feat_2 = np.load('Images_' + str(n + 1) + '.npy', allow_pickle=True)[:250]
        Target = np.load('Targets_' + str(n + 1) + '.npy', allow_pickle=True)[:250]
        Feature_1 = np.reshape(Feat_1, (Feat_1.shape[0], Feat_1.shape[1] * Feat_1.shape[2]))
        Feature_2 = np.reshape(Feat_2, (Feat_2.shape[0], Feat_2.shape[1] * Feat_2.shape[2] * Feat_2.shape[3]))
        Feat = np.concatenate((Feature_1, Feature_2), axis=1)
        BestSol = np.load('BestSol_CLS.npy', allow_pickle=True)[n]
        K = 5
        Per = 1 / 5
        Perc = round(Feat.shape[0] * Per)
        eval = []
        for i in range(K):
            Eval = np.zeros((10, 12))
            Test_Data = Feat[i * Perc: ((i + 1) * Perc), :]
            Test_Target = Target[i * Perc: ((i + 1) * Perc), :]
            test_index = np.arange(i * Perc, ((i + 1) * Perc))
            total_index = np.arange(Feat.shape[0])
            train_index = np.setdiff1d(total_index, test_index)
            Train_Data = Feat[train_index, :]
            Train_Target = Target[train_index, :]
            for j in range(BestSol.shape[0]):
                sol = np.round(BestSol[j, :]).astype(np.int16)
                Eval[j, :], pred_0 = Model_HC_AEB7_SAT(Feat_1, Feat_2, Target, sol=sol)
            Eval[5, :], pred_1 = Model_VGG16(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[6, :], pred_2 = Model_ANN(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[7, :], pred_3 = Model_CNN(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[8, :], pred_4 = Model_HC_AEB7_SAT(Feat_1, Feat_2, Target)
            Eval[9, :] = Eval[4, :]
            eval.append(Eval)
        EVAL_ALL.append(eval)
    np.save('Eval_All_Fold_error.npy', np.asarray(EVAL_ALL))


plot_conv()
Plot_Kfold_ERROR()
Plot_batchsize_error()

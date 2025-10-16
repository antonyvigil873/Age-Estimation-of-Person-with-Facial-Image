import numpy as np
from Evaluate_error import evaluate_error
from Global_Vars import Global_Vars
from Model_HC_AEB7_SAT import Model_HC_AEB7_SAT


def objfun_cls(Soln):
    Feat = Global_Vars.Feat_1
    Image = Global_Vars.Feat_2
    Tar = Global_Vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        learnper = round(Image.shape[0] * 0.75)
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            Test_Target = Tar[learnper:, :]
            Eval, pred = Model_HC_AEB7_SAT(Feat, Image, Tar, sol=sol)
            Eval = evaluate_error(pred, Test_Target)
            Fitn[i] = (1 / Eval[11]) + Eval[2]  # Accuracy + RMSE
        return Fitn
    else:
        learnper = round(Image.shape[0] * 0.75)
        sol = np.round(Soln).astype(np.int16)
        Test_Target = Tar[learnper:, :]
        Eval, pred = Model_HC_AEB7_SAT(Feat, Image, Tar, sol=sol)
        Eval = evaluate_error(pred, Test_Target)
        Fitn = (1 / Eval[11]) + Eval[2]  # Accuracy + RMSE
        return Fitn

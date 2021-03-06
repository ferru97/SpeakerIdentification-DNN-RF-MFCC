"""
get Features from training audio samples


Author: Alessandro Gerardi

"""

from GMM import GMM


def Training_feature_Mean(output_wavefile):
    Training_info = GMM(32, output_wavefile)

    Mean_training = Training_info.GMM_Model_Mean()

    return Mean_training


def Training_feature_Weight(output_wavefile):
    Training_info = GMM(32, output_wavefile)

    Weight_training = Training_info.GMM_Model_Weight()

    return Weight_training


def Training_feature_Covar(output_wavefile):
    Training_info = GMM(32, output_wavefile)

    Covar_training = Training_info.GMM_Model_Covar()

    return Covar_training

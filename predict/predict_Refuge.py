import config
from utils.predict_functions import *
from dataset.my_refuge_datasets import *
from models.UMLNet import *
import torch.backends.cudnn as cudnn
from utils.tool_functions import *

# ---- ignore all warnings ---- #
warnings.filterwarnings("ignore")

# ---- Setting a argparse and add var ---- #
model_urls = '../train/save/checkpoints/Model_results/0_uml_refuge.pt'  # Model pre train urls

# ---- Setting cuda environment ---- #
os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU_NUM_NETFINAL_REFUGE
cudnn.enabled = True
cudnn.benchmark = True


def predictNetUml():
    # ---- set data loader or just one image ---- #
    test_loader = torch.utils.data.DataLoader(
        JointRefugeDataset(path_refuge_data=config.REFUGE_DATA_FILE, data_type='test'),
        batch_size=1, num_workers=0, shuffle=False, pin_memory=False, drop_last=True)
    print("test image num:", len(test_loader))

    if not os.path.isdir(config.SEG_RESULT_PATH_OF_TEST):
        os.mkdir(config.SEG_RESULT_PATH_OF_TEST)

    # ---- load trained model ---- #
    model = UMLNet(config).cuda()
    model.load_state_dict(torch.load(model_urls))
    f_path = 'test_output.log'
    logfile = open(f_path, 'a')
    # ---- predict and get index ---- #
    result = predict_UML_REFUGE(test_data_loader=test_loader,
                                model=model,
                                save_seg_path=config.SEG_RESULT_PATH_OF_TEST,
                                noise_condition="Gaussian",  # saltPepper
                                gaussian_noise_sigma=0.0)
    # cls
    [AUC, ACC, KAPPA, SENS, F1] = result['cls']
    line = "[Cls]: AUC = %f, ACC = %f, KAPPA = %f, SENS = %f, F1 = %f" % (AUC, ACC, KAPPA, SENS, F1)
    print_f(line, f=logfile)

    # seg
    [DICE1, ASSD1, DICE2, ASSD2] = result['seg']
    line = "[Seg]: DICE1 = %f, ASSD1 = %f, DICE2 = %f, ASSD2 = %f" % (DICE1, ASSD1, DICE2, ASSD2)
    print_f(line, f=logfile)


if __name__ == '__main__':
    predictNetUml()

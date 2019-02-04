from utils import save_inps, save_st, save_inps_rcc, save_new_data
import os
data_fol = '../data'
if not (os.path.isdir(data_fol)):
	os.mkdir(data_fol)
save_st()

#calls loadSt2 which reads data from ['Penn_renal_tumor_segment_relabeled', 'TCGA', 'Xiangya Second Hospital', 'Hunan People Hospital', 'Penn renal tumor2']
#saves '../data/st1', '../data/st2'

save_inps()
save_inps_rcc()
save_new_data()

#takes in st via loadSt and saves '../data/inps'

#inps goes into the files....

'../data/inps_RCC'
'../data/inps'
'../data/inps_new_data_Fuhrman_cc.pkl'
'../data/inps_new_data_ISUP_cc.pkl'
'inps_new_data_Fuhrman'
'inps_new_data_ISUP'
# Fuhrman_repo
This repo contains the code needed to run the data loading, preprocessing, as well as training of the model, then generation of figures based off of said model. The patient data is not publicly included for privacy reasons. However the our trained models weights are included for some models.

Data is read from other folders and saved into intermediate data pickles in a data folder (../data/)


save_inps saves all input data in ../data/ which is 
['st1.pkl', 'st2.pkl', 'inps_RCC.pkl', 'inps.pkl', 'inps_new_data_Fuhrman_cc.pkl', 'inps_new_data_ISUP_cc.pkl', 'inps_new_data_Fuhrman.pkl', 'inps_new_data_ISUP.pkl']

Other functions load data from. this source

weights are similarly saved to an external ../h5 folder
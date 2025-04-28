
"""
- config_00.path_addedID_csv = path per csv con "patient_id" column
- I'd like to stratify this dataset per patient and maintaining label's proportion. 
Example of db rows:  
patient_id,dish,well,dish_well,maternal age,sperm quality,mezzo di coltura,PN,BLASTO NY,eup_aneup,LB,video_start,tPNa,tPNf,t2,t3,t4,t5,t6,t7,t8,t9+,tM,tSB,tB,tEB,t-biopsy,PN1a,PN2a,PN3a,altri Pna,PN1f,PN2f,PN3f,altri PNf,CP
54,D2013.02.19_S0675_I141,1,D2013.02.19_S0675_I141_1,37,OAT,quinns,2PN,1,euploide,non trasferito,0.174973888888888,11.9302202777777,22.9347619444444,26.1856647222222,36.9391922222222,38.4396861111111,50.5759361111111,51.82621,52.8273586111111,55.3284655555555,71.1111222219444,96.3776113888889,103.639233888888,111.894108611111,114.530137221944,138.374,12.1802797222222,12.1802797222222,-,-,22.6847,22.6847,-,-,2
54,D2013.02.19_S0675_I141,2,D2013.02.19_S0675_I141_2,37,OAT,quinns,2PN,1,euploide,non trasferito,0.177280555555555,7.68124611111111,24.1874161111111,27.1883522222222,37.6918958336111,39.9426052777777,51.0784305555555,53.5799433333333,68.3431222222222,83.4360927777777,83.6861205555555,95.8774341666666,102.891423055555,108.144410555555,113.7448325,138.376,7.68124611111111,8.681685,-,-,24.1874161111111,23.9373102775,-,-,2
55,D2013.03.09_S0695_I141,1,D2013.03.09_S0695_I141_1,40,normo,quinns,2PN,0,,,0.123866111111111,6.55515611111111,23.3513833333333,26.8526080555555,37.3587008333333,39.1091313888888,51.3415333333333,52.3418949999999,77.2186305555555,77.4687319444444,77.7188869444444,89.4723702777777,105.476802222222,138.5783275,140.656773055555,-,8.30577,9.80876444472222,-,-,23.3513833333333,23.1013122222222,-,-,2
55,D2013.03.09_S0695_I141,2,D2013.03.09_S0695_I141_2,40,normo,quinns,2PN,1,Euploide,1,0.126134722222222,7.55784388888888,24.6040891666666,28.1057827777777,40.3626602780555,47.3452758333333,54.8450158336111,59.8469325,63.59806,67.1664633333333,67.4165011111111,67.6666069441666,101.728255,120.279152222222,122.587878333333,138.344,9.05910805555555,9.81127305555555,-,-,24.3540525,24.3540525,-,-,2
55,D2013.03.09_S0695_I141,3,D2013.03.09_S0695_I141_3,40,normo,quinns,2PN,0,,,0.128466666666666,7.81043805555555,20.3546997222222,23.8564022222222,24.8564694444444,55.0975980555555,61.5998802777777,107.982258333333,108.232371944444,119.281269999999,119.531315277777,119.781480833333,-,-,-,-,8.31070055555555,8.31070055555555,-,-,20.1046552777777,20.1046552777777,-,-,2

Stratfication is used to create balanced train-val and test.
Then, I want to use the 15% of the training as a subset for the fine tuning (for optical flow algorithms parameters and for models' hyperparameters)
This subset creation has to be balanced as the first split in training-val-test.
The optimization will be done splitting this 15% subsets into 75% training and 25% validation (balanced with the same criterion)

After the optimization, with the best parameters I want to extract optical flow of the entire dataset, train models on the train selecting the best on the validation and then test everything on the test.



These "dish_well" are also the names of the directory of the blastocyst video time-lapse:
- config.user_paths.path_BlastoData (="/home/phd2/Scrivania/CorsoData/blastocisti")
"blastocisti" is the full dataset, and has 2 subfolders: "blasto" and "no_blasto". In each subfolder there are several directories whose names are the id "dish_well"
- I'd like to create other 2 main folders: the first one has to be called "test"


"""



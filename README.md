# You need run main.py from pixor_code it is my main folder in pycharm
## I remind that this is cpu version, please, add "check" functions below in your progect and change train_model(self, data_loaders, period) in addition_net_structures like I have to make it gpu version
### Next time I will check all unnecessary folders and delete them, sorry.

### HomeWork
Please, check this functions:
1) get_learning_data in kitti_datagen_exp..
2) filter_annos in kitti_datagen_exp..
3) get_anno_without_file in kitti_datagen_exp..


I change get_learning_data in kitti_datagen_exp...
Now it receive:
1) param index: index of cloud and anno which you want to aug
2) param number_of_aug: number of augment data
3) param return_clouds: key "return_clouds" if True return optional list of clouds for visualisation
return: list_clouds (if return_clouds == True), list_grid, list_output_class, list_output_reg, list_annos



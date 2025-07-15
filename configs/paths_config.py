dataset_paths = {
	'celeba_train': '',
	'celeba_test': './data/celeba_hq/test/',
	'cars_test': './data/Stanford_Cars/cars_test',
	'ffhq': './data/FFHQ_1024/',
	'cars': './data/Stanford_Cars/cars_train',

}

model_paths = {
	'stylegan_ffhq': './pretrained_models/stylegan2-ffhq-config-f.pt',
	'stylegan_cars': './pretrained_models/stylegan2-cars-config-f.pt',
	'ir_se50': './pretrained_models/model_ir_se50.pth',
	'parsing_net': './pretrained_models/parsing.pth',
	'circular_face': 'pretrained_models/CurricularFace_Backbone.pth',
	'mtcnn_pnet': 'pretrained_models/mtcnn/pnet.npy',
	'mtcnn_rnet': 'pretrained_models/mtcnn/rnet.npy',
	'mtcnn_onet': 'pretrained_models/mtcnn/onet.npy',
	'shape_predictor': 'pretrained_models/shape_predictor_68_face_landmarks.dat'
}

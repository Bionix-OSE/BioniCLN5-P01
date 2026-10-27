from ultralytics import YOLO
#from os import environ
#environ["CUDA_VISIBLE_DEVICES"] = "Please set this in your .bashrc"

if __name__ == '__main__': # Needed for Python >=3.14
	# Be sure to configure paths correctly in ~/.config/Ultralytics/settings.json
	# Some paths here will resolve from YOUR CURRENT SHELL DIRECTORY. If needed, run the file from
	# the project's root folder, or create symlinks inside each week's folders.
	model = YOLO("workdir/yolo-models/yolo11n.pt") # pretrained checkpoint, used as a starting point
	model.train( # Make sure dataset
		data="datasets/tkh-mth2k2/tkh-mth2k2.yaml",
		epochs=20,
		imgsz=640,
		single_cls=True, # collapse to one class even if labels weren't pre-converted
		batch=8,
		patience=10,
		project="workdir", # whatever you named here will be nested under 'runs/detect' in your specified "runs_dir" path
		name="week3-trained-mth",
		device=0
	)

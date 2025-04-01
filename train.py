from ultralytics import YOLO

# Load a pretrained model
model = YOLO("yolo11n.pt")

# Train the model on your custom dataset
model.train(data="dataset.yaml", 
            epochs= 150, 
            imgsz= [1280, 720], 
            batch = 8, 
            device = 0, 
            lr0 = 0.005, # Compromis entre 0.01 et 0.001 pour une convergence stable sans risque de divergence trop agressive. 0.001 trés stable mais la convergence est très lente. 0.01 Convergence rapide mais risque d'oscilation
            lrf = 0.01 ,  # Réduit fotement le lr en fin de training, mais peut rendre le taux final trop faible et stopper learning en phase finale 
            optimizer = "SGD", # Bonne généralisation et une stabilité pour gros dataset, mais nécessite un tuning comme l'ajout de momentum et peut conveger lentement
            momentum = 0.9, 
            weight_decay =0.0005, # 0.001 renforce la régularisation qui est utile pour nano, mais peut introduire underfitting en cas de régularisation trop forte 
            augment = True,
            mosaic=0.5, 
            scale=0.3, 
            patience=20, 
            freeze=0
 )

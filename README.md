# Object Detection – Cars (PyTorch)

Projekt zaliczeniowy z zakresu detekcji obiektów graficznych.

## Zakres projektu
- przygotowanie i etykietowanie własnego zbioru danych (PascalVOC)
- implementacja własnej klasy Dataset w PyTorch
- porównanie 5 modeli TorchVision
- fine-tuning modelu Faster R-CNN ResNet50
- analiza ilościowa i jakościowa wyników
- interaktywna wizualizacja detekcji

## Modele
- Faster R-CNN ResNet50 (pretrained)
- RetinaNet ResNet50
- SSD300 VGG16
- FCOS ResNet50
- Faster R-CNN MobileNet
- Faster R-CNN ResNet50 (fine-tuned)

## Instalacja zależności
```bash
pip install -r requirements.txt

## Uruchamianie
```bash
python test_models.py
python train.py
python compare_models_viewer.py

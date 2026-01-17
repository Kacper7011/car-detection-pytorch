# Object Detection – Cars (PyTorch)

Projekt zaliczeniowy z przedmiotu **Metody numeryczne**, dotyczący detekcji obiektów graficznych z wykorzystaniem biblioteki PyTorch.

## Zakres projektu
- przygotowanie i etykietowanie własnego zbioru danych (PascalVOC)
- implementacja własnej klasy Dataset w PyTorch
- porównanie 5 modeli TorchVision
- fine-tuning modelu Faster R-CNN ResNet50
- analiza wyników oraz wizualizacja detekcji

## Modele
- Faster R-CNN ResNet50
- RetinaNet ResNet50
- SSD300 VGG16
- FCOS ResNet50
- Faster R-CNN MobileNet
- Faster R-CNN ResNet50 (fine-tuned)

## Model fine-tuned
Wytrenowany model Faster R-CNN ResNet50 nie jest przechowywany bezpośrednio w repozytorium ze względu na rozmiar pliku.  
Wagi modelu dostępne są pod adresem:

https://1drv.ms/f/c/f17a058064cc30a5/IgDjzZgmRuVfQqSY06xwSQyZATFQYUDsO7APtQxonGqvOHU?e=eZdMNK

## Zbiór obrazów
Zbiór zdjęć użytych do etykietowania i testów dostępny jest pod adresem:

https://anselbpl-my.sharepoint.com/:f:/g/personal/21560_student_ans-elblag_pl/IgAKJB_Wa0K8QqCW4Z3f19pGAVke5rv99RFKlyjvUDJF-Sg?e=btfTyo

## Odtworzenie projektu

Aby odtworzyć projekt na własnym środowisku, należy przygotować zbiór danych oraz wykonać kilka kroków wstępnych.

### Przygotowanie danych
1. Należy umieścić minimum 300 obrazów w katalogu:
`dataset/images`

2. Zdjęcia powinny przedstawiać sceny drogowe, na których występują samochody (klasa etykietowana w projekcie).

### Obróbka zbioru danych (folder `utilities`)
Do przygotowania zbioru danych wykorzystano własne skrypty pomocnicze:

- `rename_images.py`  
  Ujednolica nazwy plików obrazów (np. car_001.png, car_002.png).

- `create_empty_xmls.py`  
  Tworzy puste pliki XML (PascalVOC) dla obrazów, które nie zawierają obiektów.

- `validate_annotations.py`  
  Sprawdza poprawność adnotacji (zgodność obrazów z plikami XML).

- `split_dataset.py`  
  Dzieli zbiór na części: train / val / test i zapisuje podział w plikach tekstowych.

Po wykonaniu tych kroków zbiór danych jest gotowy do użycia w PyTorch.

## Główne pliki projektu
- `dataset_pytorch.py` – implementacja własnej klasy Dataset kompatybilnej z PyTorch
- `train.py` – fine-tuning modelu Faster R-CNN ResNet50
- `test_models.py` – porównanie modeli na zbiorze testowym
- `compare_models_viewer.py` – wizualne porównanie modelu Faster R-CNN ResNet50 przed i po fine-tuningu
- `visualize_detection.py` – wizualizacja detekcji obiektów dla wybranego modelu
- `utils.py` – funkcje pomocnicze
- `requirements.txt` – lista zależności projektu

### (Opcjonalnie) środowisko wirtualne
Zalecane jest uruchamianie projektu w środowisku wirtualnym Python:

```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
pip install -r requirements.txt

## Licencja
Projekt udostępniony jest na licencji **MIT**. Szczegóły znajdują się w pliku `LICENSE`.

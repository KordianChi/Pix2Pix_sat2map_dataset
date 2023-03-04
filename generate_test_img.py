import torch
from sat2map_model import Generator
from PIL import Image
from torchvision.transforms import transforms

# wczytywanie modelu PyTorch z pliku
model = torch.load('model_pix2pix_100', map_location=torch.device('cpu'))

# tworzenie instancji klasy Generator z wczytanego modelu
generator = Generator()
generator.load_state_dict(model.state_dict())

# przesyłanie modelu z GPU na CPU
generator.to(torch.device('cpu'))

# wczytywanie obrazu
image = Image.open('ex_sat.png')

transform_sat = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((128, 128)),
                                    transforms.Normalize(mean=[0.0, 0.0, 0.0],
                                                         std=[1.0, 1.0, 1.0]),
                                   ])

img = transform_sat(image)

img = img.reshape((1, 3, 128, 128))

# przetwarzanie obrazu i zapisywanie wyniku predykcji
output_image = generator(img)
output_image = output_image[0]  # wybierz pierwszy element
output_image = transforms.ToPILImage()(output_image.cpu())
output_image.show()
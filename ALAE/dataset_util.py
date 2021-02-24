from io import BytesIO
import torchvision as tv, torchvision.transforms as tr
import lmdb
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pickle


class MultiResolutionDataset(Dataset):
	def __init__(self, path, transform, resolution=8):
		self.env = lmdb.open(
			path,
			max_readers=32,
			readonly=True,
			lock=False,
			readahead=False,
			meminit=False,
		)

		if not self.env:
			raise IOError('Cannot open lmdb dataset', path)

		with self.env.begin(write=False) as txn:
			self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

		self.resolution = resolution
		self.transform = transform

	def __len__(self):
		return self.length

	def __getitem__(self, index):
		with self.env.begin(write=False) as txn:
			key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
			img_bytes = txn.get(key)

		buffer = BytesIO(img_bytes)
		img = Image.open(buffer)
		img = self.transform(img)

		return img



if __name__ == "__main__":
	transform = tr.Compose(
		[
			# tr.Resize(32),
			# tr.RandomHorizontalFlip(),
			tr.ToTensor(),
			tr.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
		]
	)
	dataset = MultiResolutionDataset("/media/cchen/StorageDisk/yzhao/gen/EBM/short-run/utils/test_label/", transform)
	loader = DataLoader(dataset, shuffle=True, batch_size=32, num_workers=1, drop_last=True, pin_memory=True)
	data_loader = iter(loader)
	real_image, label = next(data_loader)
	print(type(real_image), label)
	print('eee')

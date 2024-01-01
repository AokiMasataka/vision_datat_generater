import argparse
import vision_data


def main():
	parser = argparse.ArgumentParser(description='このプログラムの説明（なくてもよい）')

	parser.add_argument('-e', '--export', type=str, default='./export')
	parser.add_argument('-m', '--mask', default=False, action='store_true')
	parser.add_argument('-b', '--bbox', default=False, action='store_true')
	parser.add_argument('-s', '--size', type=int, default=256)
	parser.add_argument('-n', '--num_samples', type=int, default=128)
	args = parser.parse_args()

	vision_data.make_write_samples(
		export=args.export,
		bbox=args.bbox,
		mask=args.mask,
		size=args.size,
		num_samples=args.num_samples
	)


if __name__ == '__main__':
	main()

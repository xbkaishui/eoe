import torchvision.transforms as transforms
import torch 
from eoe.main import default_argsparse, create_trainer, load_setup

if __name__ == '__main__':
    def modify_parser(parser):
        parser.set_defaults(
            comment='{obj}_chip_{admode}_E{epochs}',
            objective='clip',
            dataset='chip',
            # todo change for test, default is emnist
            # oe_dataset=None,
            oe_dataset='chip',
            # todo change for test, default is 80
            epochs=40,
            learning_rate=2e-5,
            weight_decay=1e-3,
            milestones=[50, 60, 70, 75],
            batch_size=40,
            devices=[0],
            classes=None,
            # todo change for test default is 10
            iterations=2,
        )
    args = default_argsparse(
        lambda s: f"{s} This specific script comes with a default configuration for training CLIP with MNIST.", modify_parser
    )
    args.comment = args.comment.format(obj=args.objective, admode=args.ad_mode, epochs=args.epochs)
    train_transform = transforms.Compose([
        transforms.Resize((23, 23)),
        # transforms.RandomCrop(12, padding=3),
        transforms.RandomHorizontalFlip(),
        'clip_pil_preprocessing',
        transforms.ToTensor(),
        # transforms.Lambda(lambda x: x + 0.001 * torch.randn_like(x)),
        'clip_tensor_preprocessing'
    ])
    val_transform = transforms.Compose([])
    snapshots, continue_run = load_setup(args.load, args, train_transform, val_transform)
    model = None

    print('Program started with:\n', vars(args))
    trainer = create_trainer(
        args.objective, args.comment, args.dataset, args.oe_dataset, args.epochs, args.learning_rate, args.weight_decay,
        args.milestones, args.batch_size, args.ad_mode, args.devices, model, train_transform, val_transform,
        continue_run=continue_run
    )

    trainer.run(args.classes, args.iterations, snapshots)

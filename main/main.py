from dataloader import *
from trainer_helper import *
from eval_helper import *
from trainer import *


def main():
    parser = make_args_parser()
    args = parser.parse_args()

    # Initialize trainer
    trainer = Trainer(args, console=True)

    # Run training
    trainer.run_train()

    # Final evaluation
    trainer.run_eval_final()

    # Spatial encoder evaluation
    val_preds = trainer.run_eval_spa_enc_only(
        eval_flag_str="LocEnc ", load_model=True)


if __name__ == "__main__":
    main()

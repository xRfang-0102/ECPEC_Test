# -*- coding: utf-8 -*-
"""加载指定 checkpoint，在 dev/test 上评估。用法: python eval.py --load_checkpoint best_model.pt"""
import os, sys, torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config, get_model_config
from data_loader import create_feature_data_loaders
from models import Model
from loss import PairLoss
from utils import set_seed, setup_logging
from train import evaluate


def main():
    config = Config()
    args = config.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger = setup_logging(args.log_dir, args.dataset, 'eval_only')

    train_loader, test_loader, dev_loader, _ = create_feature_data_loaders(
        dataset_name=args.dataset,
        feature_dir=args.feature_dir,
        batch_size=args.batch_size,
        pred_future_cause=args.pred_future_cause,
        use_emocate=args.use_emocate,
        eval_max_pair_distance=args.eval_max_pair_distance,
        train_max_pair_distance=args.train_max_pair_distance,
        max_doc_len=args.max_doc_len,
    )

    model = Model(get_model_config(args)).to(device)

    ckpt_path = os.path.join(args.save_dir, args.load_checkpoint or 'best_model.pt')
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    logger.info(f"Loaded {ckpt_path} (epoch={ckpt.get('epoch', '?')})")

    criterion = PairLoss()

    if dev_loader is not None:
        logger.info("=== Dev ===")
        m = evaluate(model, dev_loader, device, logger, args, pair_criterion=criterion)
        logger.info(f"Dev  Pair P/R/F1: {m['pair_precision']:.4f}/{m['pair_recall']:.4f}/{m['pair_f1']:.4f}")

    logger.info("=== Test ===")
    m = evaluate(model, test_loader, device, logger, args, pair_criterion=criterion)
    logger.info(f"Test Pair P/R/F1: {m['pair_precision']:.4f}/{m['pair_recall']:.4f}/{m['pair_f1']:.4f}")


if __name__ == '__main__':
    main()
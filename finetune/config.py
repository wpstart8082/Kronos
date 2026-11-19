import os

class Config:
    """
    Configuration class for the entire project.
    """

    def __init__(self):
        # =================================================================
        # Data & Feature Parameters
        # =================================================================
        # Index (14 major CN indices) qlib dump exported from MongoDB
        # (see finetune/data_prep/export_indices_to_qlib.py), mounted in the
        # kronos-finetune container.
        self.qlib_data_path = "/workspace/data/qlib_data/cn_data_indices"
        self.instrument = 'all'

        # Overall time range for data loading from Qlib.
        # Values below reflect the audited MongoDB coverage; the test slice
        # starts ~100 trading days before the backtest so the first backtest
        # day already has full lookback context (lookback 90 + predict 10).
        self.dataset_begin_time = "2015-01-01"
        self.dataset_end_time = '2026-11-18'

        # Sliding window parameters for creating samples.
        self.lookback_window = 90  # Number of past time steps for input.
        self.predict_window = 10  # Number of future time steps for prediction.
        self.max_context = 512  # Maximum context length for the model.

        # Features to be used from the raw data.
        self.feature_list = ['open', 'high', 'low', 'close', 'vol', 'amt']
        # Time-based features to be generated.
        self.time_feature_list = ['minute', 'hour', 'weekday', 'day', 'month']

        # =================================================================
        # Dataset Splitting & Paths
        # =================================================================
        # Note: The validation/test set starts earlier than the training/validation set ends
        # to account for the `lookback_window`.
        self.train_time_range = ["2015-01-01", "2023-12-31"]
        self.val_time_range = ["2023-10-01", "2025-06-30"]
        self.test_time_range = ["2025-02-01", "2026-11-18"]
        # NOTE: backtest end must stay BEFORE the last calendar day - qlib's
        # executor reads calendar[index+1] for the step end and crashes with
        # IndexError when end_time == last trading day (2026-11-18).
        self.backtest_time_range = ["2025-07-01", "2026-11-11"]

        # Directory to save the processed, pickled datasets (absolute so it is
        # independent of the CWD the train scripts are launched from).
        self.dataset_path = "/workspace/data/processed_datasets_indices"

        # =================================================================
        # Training Hyperparameters
        # =================================================================
        self.clip = 5.0  # Clipping value for normalized data to prevent outliers.

        self.epochs = 30
        self.log_interval = 100  # Log training status every N batches.
        self.batch_size = 50  # Batch size per GPU.

        # Number of samples to draw for one "epoch" of training/validation.
        # This is useful for large datasets where a true epoch is too long.
        # Index data: the train pool holds only ~28k windows (vs 7.6M for the
        # stock run), so one "epoch" samples ~54% of the pool.
        self.n_train_iter = 300 * self.batch_size
        self.n_val_iter = 100 * self.batch_size

        # Learning rates for different model components. Continued fine-tuning
        # from the stock-tuned checkpoints -> 4x lower than first-stage training
        # to avoid catastrophic forgetting.
        self.tokenizer_learning_rate = 5e-5
        self.predictor_learning_rate = 1e-5

        # Gradient accumulation to simulate a larger batch size.
        self.accumulation_steps = 1

        # AdamW optimizer parameters.
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.95
        self.adam_weight_decay = 0.1

        # Miscellaneous
        self.seed = 100  # Global random seed for reproducibility.

        # =================================================================
        # Experiment Logging & Saving
        # =================================================================
        self.use_comet = False # Set to False if you don't want to use Comet ML
        self.comet_config = {
            # It is highly recommended to load secrets from environment variables
            # for security purposes. Example: os.getenv("COMET_API_KEY")
            "api_key": "YOUR_COMET_API_KEY",
            "project_name": "Kronos-Finetune-Demo",
            "workspace": "your_comet_workspace" # TODO: Change to your Comet ML workspace name
        }
        self.comet_tag = 'finetune_demo'
        self.comet_name = 'finetune_demo'

        # Base directory for saving model checkpoints and results (absolute,
        # lives in the kronos-finetune-outputs docker volume).
        self.save_path = "/workspace/outputs/models"
        self.tokenizer_save_folder_name = 'index_tokenizer_v1'
        self.predictor_save_folder_name = 'index_predictor_v1'
        self.backtest_save_folder_name = 'index_backtest_v1'

        # =================================================================
        # Smoke Test Switch (env FT_SMOKE=1)
        # =================================================================
        # Shrinks the run to 1 epoch with a tiny sample budget so the whole
        # pipeline can be validated end-to-end in minutes. Must run BEFORE the
        # derived paths below so the *_smoke folders are used everywhere.
        if os.environ.get("FT_SMOKE") == "1":
            self.epochs = 1
            self.n_train_iter = 20 * self.batch_size
            self.n_val_iter = 5 * self.batch_size
            self.tokenizer_save_folder_name += "_smoke"
            self.predictor_save_folder_name += "_smoke"
            self.backtest_save_folder_name += "_smoke"

        # Path for backtesting results.
        self.backtest_result_path = "/workspace/outputs/backtest_results"

        # =================================================================
        # Model & Checkpoint Paths
        # =================================================================
        # Transfer fine-tuning (plan B): start from the stock-tuned v1
        # checkpoints, which already learned the A-share daily-K distribution.
        self.pretrained_tokenizer_path = "/workspace/outputs/models/as_tokenizer_v1/checkpoints/best_model"
        self.pretrained_predictor_path = "/workspace/outputs/models/as_predictor_v1/checkpoints/best_model"

        # Paths to the fine-tuned models, derived from the save_path.
        # These will be generated automatically during training.
        self.finetuned_tokenizer_path = f"{self.save_path}/{self.tokenizer_save_folder_name}/checkpoints/best_model"
        self.finetuned_predictor_path = f"{self.save_path}/{self.predictor_save_folder_name}/checkpoints/best_model"

        # =================================================================
        # Plan-A Variant Switch (env FT_PLAN_A=1)
        # =================================================================
        # Transfer-free index fine-tune: start the predictor from the ORIGINAL
        # Kronos-base and keep the ORIGINAL Kronos-Tokenizer-base frozen (the
        # tokenizer stage is skipped entirely). Ultra-conservative schedule for
        # the ~28k-window index pool: 5 epochs at LR 1e-6 - the stock-lineage
        # transfer run (index_*_v1) overfit from epoch 1, and the zero-shot
        # original turned out to be the strongest index baseline.
        if os.environ.get("FT_PLAN_A") == "1":
            self.epochs = 5
            self.predictor_learning_rate = 1e-6
            self.pretrained_predictor_path = "/workspace/models/Kronos-base"
            self.tokenizer_save_folder_name = 'index_tokenizer_fromscratch_v1'
            self.predictor_save_folder_name = 'index_predictor_fromscratch_v1'
            self.backtest_save_folder_name = 'index_backtest_fromscratch_v1'
            # Predictor training loads the tokenizer from finetuned_tokenizer_path
            # -> point it at the frozen original instead of a fine-tuned one.
            self.finetuned_tokenizer_path = "/workspace/models/Kronos-Tokenizer-base"

        # =================================================================
        # Backtesting Parameters
        # =================================================================
        # Only 14 indices in the pool -> rotate a compact Top5 portfolio.
        self.backtest_n_symbol_hold = 5  # Number of symbols to hold in the portfolio.
        self.backtest_n_symbol_drop = 1  # Number of symbols to drop from the pool.
        self.backtest_hold_thresh = 5  # Minimum holding period for a stock.
        self.inference_T = 0.6
        self.inference_top_p = 0.9
        self.inference_top_k = 0
        # 3 sampled paths (avg) instead of 5: 1.67x faster full-market inference,
        # signal is a mean over paths so 3 samples is a mild variance increase.
        self.inference_sample_count = 3
        self.backtest_batch_size = 1000
        self.backtest_benchmark = self._set_benchmark(self.instrument)

    def _set_benchmark(self, instrument):
        dt_benchmark = {
            'csi800': "SH000906",
            'csi1000': "SH000852",
            'csi300': "SH000300",
            'all': "SH000300",  # full A-share market benchmarks against CSI300
        }
        if instrument in dt_benchmark:
            return dt_benchmark[instrument]
        else:
            raise ValueError(f"Benchmark not defined for instrument: {instrument}")

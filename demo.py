from src.pipeline.train_pipeline import TrainPipeline


if __name__ == "__main__":
    print("=" * 60)
    print("Starting Spam-Ham Training Pipeline")
    print("=" * 60)

    TrainPipeline().run_pipeline()

    print("=" * 60)
    print("Training Pipeline Completed")
    print("=" * 60)

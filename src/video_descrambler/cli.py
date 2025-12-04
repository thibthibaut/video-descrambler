from pathlib import Path
import typer
from typing_extensions import Annotated
from . import VideoDescrambler

app = typer.Typer(help="Video Descrambler - Reorder scrambled video frames")


@app.command()
def descramble(
    input_video: Annotated[Path, typer.Argument(help="Path to scrambled input video")],
    output_video: Annotated[
        Path, typer.Option("--output", "-o", help="Path for descrambled output video")
    ],
):
    """
    Descramble a video with randomized frames

    Example:
        video-descrambler scrambled.mp4 -o fixed.mp4
    """

    # Validate input
    if not input_video.exists():
        typer.echo(f"Error: Input video '{input_video}' not found", err=True)
        raise typer.Exit(code=1)

    # Create descrambler with defaults
    descrambler = VideoDescrambler.pixel_based()

    typer.echo(f"Descrambling {input_video}...")
    try:
        descrambler.descramble(video_path=input_video, output_path=output_video)
        typer.echo(f"Descrambled video saved to {output_video}")
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()

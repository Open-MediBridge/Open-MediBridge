"""`python -m open_medi` starts the API + web UI with uvicorn."""

import uvicorn


def main() -> None:
    uvicorn.run("open_medi.main:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()

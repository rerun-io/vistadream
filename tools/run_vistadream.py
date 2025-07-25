import tyro

from vistadream.api.vistadream_pipeline import VistaDreamConfig, main

if __name__ == "__main__":
    main(tyro.cli(VistaDreamConfig, name="vista_dream"))

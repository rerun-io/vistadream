import tyro

from vistadream.api.single_img_nooutpaint import SingleImageConfig, main

if __name__ == "__main__":
    main(tyro.cli(SingleImageConfig, name="vista_dream"))

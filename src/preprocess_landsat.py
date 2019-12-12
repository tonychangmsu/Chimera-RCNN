import src.preprocess as prep

if __name__ == "__main__":

    wd = '/data/wood-supply'
    data = prep.generate_y_table(wd=wd)
    landsat = prep.preprocess_landsat(data)
    prep.write_landsate(data, landsat)


# TODO rethink labelling, e.g. hierarchical?
class Labels:
    """
    Convenience class for label codes.
    """
    UNLABELLED = 0
    GROUND = 1
    BUILDING = 2
    SKY = 3
    CABLE = 11
    LIGHT_CABLE = 12
    TRAM_CABLE = 13
    ARMATUUR = 15
    NOISE = 99

    STR_DICT = {0: 'Unlabelled',
                1: 'Ground',
                2: 'Building',
                3: 'Sky',
                11: 'Cable',
                12: 'Light cable',
                13: 'Tram cable',
                15: 'Suspended streetlight',
                16: 'Suspended streetlight',
                17: 'Suspended streetlight',
                18: 'Suspended streetlight',
                19: 'Suspended streetlight',
                20: 'Suspended streetlight',
                99: 'Noise'}

    @staticmethod
    def get_str(label):
        return Labels.STR_DICT[label]

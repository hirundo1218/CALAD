import os

class MyPath(object):
    @staticmethod
    def db_root_dir(database=''):
        db_names = {'msl', 'smap', 'smd', 'swat'}
        assert(database in db_names)

        if database == 'msl' or database == 'smap':
            return 'datasets/MSL_SMAP'
        elif database == 'smd':
            return 'datasets/SMD'
        elif database == 'swat':
            return 'datasets/SWAT'
        else:
            raise NotImplementedError
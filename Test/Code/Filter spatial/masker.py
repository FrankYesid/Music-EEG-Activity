# Frank Y. Zapata C.
# Version 2021
# ------------------------------------------------------------------------

import re
import mne
import numpy as np


########################################################################
class Masker:
    """
    """

    # ----------------------------------------------------------------------
    def __init__(self, channels, montage_name):
        """Constructor"""
        self.template = self.get_template(montage_name)
        self.montage = mne.channels.make_standard_montage(montage_name)
        self.channels = channels

    # ----------------------------------------------------------------------
    def get_mask(self):
        """"""
        data = []
        for column in self.template:
            c = []
            for channel in self.montage.ch_names:
                for channelc in column:
                    if re.match(channelc, channel):
                        c.append(channel)
            data.append(c)

        data = [sorted(d, key=self.get_sort_value) for d in data]

        max_ = max([len(d) for d in data])
        map_ = np.array([np.pad(d, (max_ - len(d)) // 2)
                         for d in data], dtype=object)
        map_[map_ == '0'] = 0
        r = map_.shape[0]

        for i, ch in enumerate(self.channels):
            map_[map_ == ch] = i + 1

        map_ = np.array(
            [c if isinstance(c, int) else 0 for c in map_.flatten()]).reshape(r, -1)

        masq = map_[map_.sum(axis=1) != 0]
        masq = ([r[r != 0] for r in masq])
        max_w = max([len(m) for m in masq])

        masq = [m.tolist() for m in masq]
        for row in masq:
            if len(row) % 2 == 0:
                c = len(row) // 2
                row.insert(c, 0)

        masq = [np.pad(d, (max_w - len(d)) // 2) for d in masq]
        masq = np.array(masq, dtype=int)

        return masq

    # ----------------------------------------------------------------------
    @staticmethod
    def get_sort_value(v):
        """"""
        match = re.findall('([0-9]+)', v)
        if match:
            order = int(match[0])
            if int(order) % 2 != 0:
                order = -order
            if v.endswith('h'):
                if order > 0:
                    order -= 0.5
                else:
                    order += 0.5
            return order
        else:
            return 0

    # ----------------------------------------------------------------------
    def get_template(self, montage_name):
        """"""
        if montage_name == 'standard_1020':
            template = (
                ['^Fp[z0-9]{1,2}$'],
                ['^AF[z0-9]{1,2}$'],
                ['^F[z0-9]{1,2}$'],
                ['^FC[z0-9]{1,2}$', '^FT[0-9]{1,2}$'],
                ['^C[z0-9]{1,2}$', '^T[0-9]{1,2}$'],
                ['^CP[z0-9]{1,2}$', '^TP[0-9]{1,2}$'],
                ['^P[z0-9]{1,2}$'],
                ['^PO[z0-9]{1,2}$'],
                ['^O[z0-9]{1,2}$'],
                ['^I[z0-9]{1,2}$'],
            )
        # montage_ = mne.channels.make_standard_montage('standard_1020')

        elif montage_name == 'standard_1005':
            template = (
                ['^Fp[z0-9]{1,2}[h]*$'],
                ['^AFp[z0-9]{1,2}[h]*$'],
                ['^AF[z0-9]{1,2}[h]*$'],
                ['^AFF[z0-9]{1,2}[h]*$'],
                ['^F[z0-9]{1,2}[h]*$'],
                ['^FFC[z0-9]{1,2}[h]*$', '^FFT[0-9]{1,2}[h]*$'],
                ['^FC[z0-9]{1,2}[h]*$', '^FT[0-9]{1,2}[h]*$'],
                ['^FCC[z0-9]{1,2}[h]*$', '^FTT[0-9]{1,2}[h]*$'],
                ['^C[z0-9]{1,2}[h]*$', '^T[0-9]{1,2}[h]*$'],
                ['^CCP[z0-9]{1,2}[h]*$', '^TTP[0-9]{1,2}[h]*$'],
                ['^CP[z0-9]{1,2}[h]*$', '^TP[0-9]{1,2}[h]*$'],
                ['^CPP[z0-9]{1,2}[h]*$', '^TPP[0-9]{1,2}[h]*$'],
                ['^P[z0-9]{1,2}[h]*$'],
                ['^PPO[z0-9]{1,2}[h]*$'],
                ['^PO[z0-9]{1,2}[h]*$'],
                ['^POO[z0-9]{1,2}[h]*$'],
                ['^O[z0-9]{1,2}[h]*$'],
                ['^OI[z0-9]{1,2}[h]*$'],
                ['^I[z0-9]{1,2}[h]*$'],

            )

        return template


# ----------------------------------------------------------------------
def generate_mask(channels, montage_name):
    """"""
    masker = Masker(channels, montage_name)
    return masker.get_mask()

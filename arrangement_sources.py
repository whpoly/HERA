"""Lightweight registry shared by the training workers and their queue."""
from pathlib import Path
from typing import NamedTuple


ROOT = Path(__file__).resolve().parent


class Source(NamedTuple):
    module: str
    schema: str
    results_folder: str

    @property
    def results(self):
        return ROOT/'results'/self.results_folder

    @property
    def logs(self):
        return ROOT/'logs'/self.schema


SOURCES = {
    'ysz': Source('ysz_arrangements','ysz_parkes_fixed_composition_gap_v1','ysz_parkes_arrangements_20261110'),
    'hzo': Source('hzo_vacancy_sites','hzo_taper_equimolar_vacancy_gap_v1','hzo_taper_vacancy_sites_20261110'),
    'pyrochlore': Source('pyrochlore_arrangements','pyrochlore_equimolar_B_order_gap_v1','pyrochlore_equimolar_arrangements_20261110'),
    'nasicon': Source('nasicon_vacancy_order','nasicon_park_Na2_vacancy_order_gap_v1','nasicon_park_Na2_vacancies_20261110'),
    'nb_split': Source('niobium_split_vacancies','nb_split_vacancy_relaxed_gap_v1','nb_split_vacancies_20261110'),
    'nb_h8': Source('niobium_hydride_order','niobium_half_hydride_relaxed_gap_v1','nb_hydride_H8_arrangements_20261111'),
    'feni': Source('feni_vacancy_order','feni_fisher_order_vacancy_gap_v1','feni_fisher_vacancy_order_20261111'),
    'ceria4': Source('ceria_four_vacancies','ceria_111_four_oxygen_vacancies_gap_v1','ceria_111_four_vacancies_20261111'),
}

import os
import argparse
import yaml
import logging
from madminer.delphes import DelphesReader
from your_observables import define_observables

logging.basicConfig(
    format='%(asctime)s %(name)-20s %(levelname)-7s %(message)s',
    datefmt='%H:%M',
    level=logging.DEBUG
)

def process_events(sample_dir, setup_file, is_background, delphes_card, do_delphes, benchmark, output_dir):
    reader = DelphesReader(setup_file)

    reader.add_sample(
        hepmc_filename=f'{sample_dir}/tag_1_pythia8_events.hepmc.gz',
        lhe_filename=f'{sample_dir}/unweighted_events.lhe.gz',
        delphes_filename=None if do_delphes else f'{sample_dir}/tag_1_pythia8_events_delphes.root',
        sampled_from_benchmark=benchmark,
        is_background=is_background,
        k_factor=1.0,
        weights='lhe'
    )

    if do_delphes:
        if os.path.exists(f'{sample_dir}/tag_1_pythia8_events_delphes.root'):
            logging.warning('Delphes file already exists!')
        reader.run_delphes(
            delphes_directory='/cvmfs/sw.el7/gcc63/madgraph/3.3.1/b01/Delphes/',
            delphes_card=delphes_card,
            initial_command='module load gcc63/madgraph/3.3.1; source /cvmfs/sw.el7/gcc63/madgraph/3.3.1/b01/Delphes/DelphesEnv.sh',
            log_file=f'{sample_dir}/do_delphes.log'
        )

    define_observables(reader)

    reader.analyse_delphes_samples(delete_delphes_files=True)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'analysed_events.h5')
    reader.save(output_path)
    logging.info(f"Saved: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', required=True)
    parser.add_argument('--sample_dir', required=True)
    parser.add_argument('--do_delphes', action='store_true', default=False)
    parser.add_argument('--benchmark', required=True)

    args = parser.parse_args()

    with open(args.config_file, 'r') as file:
        config = yaml.safe_load(file)
        setup_file = os.path.join(config['main_dir'], 'setup_1D_cHGtil.h5')
        delphes_card = os.path.join(config['main_dir'], config['delphes_card'])

    is_background = 'background' in args.sample_dir

    process_events(
        sample_dir=args.sample_dir,
        setup_file=setup_file,
        is_background=is_background,
        delphes_card=delphes_card,
        do_delphes=args.do_delphes,
        benchmark=args.benchmark,
        output_dir='/project/atlas/users/sabdadin/output'
    )

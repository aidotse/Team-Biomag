n_images = 10;
magnification = 60;
output_dir = '/home/koosk/data/images/adipocyte/generated_masks/';

copyfile(['simcep_options', num2str(magnification), '.m'], 'simcep_options.m')

parfor i = 1:n_images
    [~, binary] = simcep;
    random_id = char(java.util.UUID.randomUUID);
    random_id = random_id(1:8);
    fname = ['generated_nuclei_mask_', num2str(magnification), 'x-', datestr(now, 'YYYY-mm-DD_HHMMssFFF'), '-', random_id, '.png'];
    fpath = fullfile(output_dir, fname);
    imwrite(binary, fpath);
end

delete('simcep_options.m')
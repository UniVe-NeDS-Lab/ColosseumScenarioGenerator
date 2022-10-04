function[] = convertColosseum(scenarioName)
    inFolder = "scenarios/"+scenarioName+"/colosseum";
    outFolder = "scenarios/"+scenarioName+"/matlab";
    mkdir(outFolder);
    duration = 1;

    %% Nodes Positions
    mat = readmatrix(inFolder+"/nodesPosition.csv");
    m_size = size(mat);
    nodesPositions = cell(1, m_size(1));

    for i = 1:m_size(1)
        nodesPositions{i} = [mat(i,1), mat(i,2), mat(i,3)];
    end
    save(fullfile(outFolder, 'nodesPositions'), 'nodesPositions');
    %% channel matrix
    delayMatrix = readmatrix(inFolder+'/delayMatrix.csv');
    pathlossMatrix = readmatrix(inFolder+'/pathlossMatrix.csv');

    clusteredTapMatrix = cell(m_size(1), m_size(1));
    for i = 1:m_size(1)
        for j = 1:m_size(1)
            tap.delay = [delayMatrix(i,j), 0, 0, 0];
            tap.iq = [db2mag(-pathlossMatrix(i,j)), 0, 0, 0];
            clusteredTapMatrix{i, j} = tap;
        end
    end

    save(fullfile(outFolder, 'clusteredTapMatrix'), 'clusteredTapMatrix');

    %% time stamps
    assert(duration == round(duration),...
        'duration must be an integer')
    timestamps = [0, duration*1000]; % ms

    save(fullfile(outFolder, 'timestamps'), 'timestamps');
end
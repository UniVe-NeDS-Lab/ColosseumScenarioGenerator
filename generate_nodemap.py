import requests
import configargparse
import os
import networkx as nx
from dotenv import load_dotenv
from pprint import pprint
import json
load_dotenv()
COLOSSEUM_USER = os.getenv('COLOSSEUM_USER')
COLOSSEUM_PWD = os.getenv('COLOSSEUM_PWD')


def login_colosseum():
    json_data = {
        'username': COLOSSEUM_USER,
        'password': COLOSSEUM_PWD,
    }
    print(json_data)
    response = requests.post('https://experiments.colosseum.net/api/v1/auth/login/',  json=json_data,  verify=False)
    if response.status_code != 200:
        raise Exception("Error logging in")
    for c in response.cookies:
        if c.name == 'sessionid':
            return c.value
    raise Exception("No cookie found")


def get_reservation(session_id, reservation_id):
    cookies = {
        'sessionid': session_id,
    }

    headers = {
        'Accept': 'application/json, text/plain, */*',
    }

    response = requests.get(f'https://experiments.colosseum.net/api/v1/reservations/{reservation_id}/', cookies=cookies, headers=headers, verify=False)
    if response.status_code != 200:
        raise Exception("Error getting reservation")
    return response.json()


def main(args):
    nodemap = {}
    session = login_colosseum()
    reservation = get_reservation(session, args.reservation)
    srns = sorted(reservation[0]['nodes'], key=lambda x: x['srn_id'])
    core = [x['srn_id'] for x in srns if 'oai-cn' in x['image']]
    ran = [x['srn_id'] for x in srns if 'oai-ran' in x['image']]
    print(f'Core is {core[0]}')
    print(f'RAN is {ran}')
    graph = nx.read_graphml(f'scenarios/{args.scenario_name}/graph.graphml', node_type=str)
    # TODO: check sizes of nodesubset, ran and graph.nodes
    if args.nodesubset:
        m = 0
        for ndx, n in enumerate(graph.nodes()):
            if ndx+1 in args.nodesubset:
                print(f'Node {ndx+1} to SRN {ran[m]}')
                nodemap[f'Node {ndx+1}'] = {"SRN": ran[m], "RadioA": 1, "RadioB": 2}
                m += 1
            else:
                nodemap[f'Node {ndx+1}'] = "None"
    else:
        for ndx, n in enumerate(graph.nodes()):
            if ndx+1 < len(ran):
                nodemap[f'Node {ndx+1}'] = {"SRN": ran[ndx+1], "RadioA": 1, "RadioB": 2}
            else:
                nodemap[f'Node {ndx+1}'] = "None"
    with open(f'scenarios/{args.scenario_name}/nodemap.json', 'w') as fw:
        json.dump(nodemap, fw, indent=2)


if __name__ == '__main__':
    p = configargparse.ArgParser(default_config_files=['nodemap_gen.yaml'])
    p.add('--scenario_name', required=True, type=str)
    p.add('--reservation', required=True, type=str)
    p.add('--nodesubset', required=False, type=int, action='append')
    args = p.parse_args()
    main(args)

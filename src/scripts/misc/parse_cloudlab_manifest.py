import argparse
import os
from xml.dom.minidom import parse


def main(rspec_path: str, output_path: str, keypath: str, identifier: str = "node"):
    if not os.path.exists(rspec_path):
        raise FileNotFoundError("No rspec found at " + rspec_path)

    # parse rspec document
    rspec_document = parse(rspec_path)
    rspec_nodes = rspec_document.getElementsByTagName("node")

    # Obtain relevant node information from rspec nodes
    nodes_list = []
    for node in rspec_nodes:
        node_info = dict()
        node_info["hostname"] = node.getElementsByTagName("services")[0].childNodes[1]._attrs['hostname']._value
        node_info["id"] = node.getAttributeNode("client_id").value
        node_info.update(node.getElementsByTagName("services")[0].getElementsByTagName("login")[0].attributes.items())
        node_info["shortname"] = identifier + node_info["id"].split("_")[1]
        print("Found node with settings: " + str(node_info))
        nodes_list.append(node_info)

    generate_sshconf(nodes_list, output_path, keypath)
    generate_hostname_list(nodes_list, output_path)


def generate_hostname_list(nodes: list, output: str):
    path = output + "/hostnames"
    open(path, 'w').close()
    with open(path, 'a') as file:
        for node_info in nodes:
            file.write(node_info["hostname"] + "\n")


def generate_sshconf(nodes: list, output: str, keypath: str):
    path = output + "/sshconfig"
    open(path, 'w').close()
    # write contents to file
    with open(path, 'a') as file:
        for node_info in nodes:
            file.write("Host " + node_info["shortname"] + "\n")
            file.write("Hostname " + node_info["hostname"] + "\n")
            file.write("User " + node_info["username"] + "\n")
            file.write("IdentityFile " + keypath + "\n")
            file.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Converting Cloud-Lab rspec to ssh-config file')
    parser.add_argument("--input", required=True, type=str, help="Path to manifest file")
    parser.add_argument("--output", required=True, type=str, help="Output directory where to store ssh-files")
    parser.add_argument("--identifier", required=True, type=str,
                        help="SSH-identifier to name the cluster and the instances. "
                             "Example:  'node' will give: node0, node1, ... as instance names")
    parser.add_argument("--keypath", required=True, type=str, help="Path to ssh-key")
    args = parser.parse_args()
    main(args.input, args.output, args.keypath, args.identifier)
    print("Successful!")
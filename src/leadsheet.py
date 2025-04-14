import music21
import numpy as np

# for visualization
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import copy

from hmmlearn import _hmmc

"""
known issues:
coda on All Or Nothing At All happens a bar later than it should
it is the same in musicXML, but different in irealpro app. maybe
it is an exporting bug in irealpro app

Cabin in the Sky - does not start with a repeat sign, although it should

TODO: check codas in other songs
TODO: recognize if CODA is infinite repeat, like in farmer's trust
TODO: recognize solos (as in "goodbye pork pie hat")
TODO: different form on head and on solos (Hallucinations)
TODO: parse 4x sign (as in little dancer)
TODO: parse 8x sign (Three Flowers)

I Feel Pretty, Little Dancer - weird repeat ends

Following songs does not have a repeat start at position 0, although shoud:
Cabin in the Sky, Everybody's Song But My Own, Five Brothers, Freddie Freeloader, 
Funk In Deep Freeze, Ill Wind, It Might As Well Be Spring, Lover Man,
Ol' Man River, Our Delight, P.S. I Love You, Three Flowers

to solve this, _fix() method was added to the LeadSheet class

also, still some problems with coda signs ...

----

"Diverse", "Segment": missing "2nd time ending" - adding a quick fix for this
"Zingaro (Retrato Em Branco E Preto)" -- missing coda sign, wrong segmentation
"""


class HMMNode:
    def __init__(self, num=None, figure=None, chroma=None, children=None):
        self.num = num
        self.figure = figure
        self.chroma = chroma
        self.children = children

    def emission_prob(self):
        pass


class HMMConnectorMixin:
    def build_subnodes(self):
        beat_counter = 0
        for measure in self.measures:
            measure.subnodes = []
            for j, beat_state in enumerate(measure.beat_states):
                measure.subnodes.append(
                    HMMNode(
                        num=beat_counter,
                        figure=beat_state.figure,
                        chroma=beat_state.chroma,
                        children=[],
                    )
                )
                beat_counter += 1
                if j != 0:
                    measure.subnodes[-2].children.append(measure.subnodes[-1])

    def build_subgraph(self):
        self.build_subnodes()
        for measure in self.measures:
            for key, elem in measure.children.items():
                measure.subnodes[-1].children.append(elem.subnodes[0])

    def build_adjacency_matrix(self):
        num_beats = self.measures[-1].subnodes[-1].num + 1
        adjacency_matrix = np.zeros((num_beats, num_beats))
        for measure in self.measures:
            for node in measure.subnodes:
                if node.children:
                    for child in node.children:
                        adjacency_matrix[node.num, child.num] += 1
        return adjacency_matrix

    def build_2nd_adjacency_matrix(self):
        num_beats = self.measures[-1].subnodes[-1].num + 1
        adjacency_matrix = np.zeros((num_beats, num_beats))
        for measure in self.measures:
            for node in measure.subnodes:
                if node.children:
                    for child in node.children:
                        if child.children:
                            for grandchild in child.children:
                                adjacency_matrix[node.num, grandchild.num] += 1
        return adjacency_matrix

    def build_transition_matrix(self, stay=0.1, step=0.8, skip=0.1, probs=None):
        if probs:
            stay, step, skip = probs
        num_nodes = self.measures[-1].subnodes[-1].num + 1
        zero_adjacency_matrix = np.eye(num_nodes)
        first_adjacency_matrix = self.build_adjacency_matrix()
        first_adjacency_matrix = first_adjacency_matrix / first_adjacency_matrix.sum(
            axis=1, keepdims=True
        )
        second_adjacency_matrix = self.build_2nd_adjacency_matrix()
        second_adjacency_matrix = second_adjacency_matrix / second_adjacency_matrix.sum(
            axis=1, keepdims=True
        )
        transition_matrix = (
            stay * zero_adjacency_matrix
            + step * first_adjacency_matrix
            + skip * second_adjacency_matrix
        )
        return transition_matrix

    def build_node_chroma_matrix(self):
        num_beats = self.measures[-1].subnodes[-1].num + 1
        node_chroma_matrix = np.zeros((num_beats, 12))
        for measure in self.measures:
            for node in measure.subnodes:
                node_chroma_matrix[node.num] = node.chroma["pitches"]
        return node_chroma_matrix

    def calculate_normalization_constant(self):
        chroma_matrix = np.zeros((4096, 12))
        # each column is a binary representation of an integer in range 0-4095
        for i in range(4096):
            chroma_matrix[i] = np.array([int(x) for x in f"{i:012b}"])
        distances = 1 / (1 + np.exp(chroma_matrix.sum(axis=1)))
        return np.sum(distances)

    @staticmethod
    def compute_log_likelihood(
        node_chroma_matrix, sequence, transposition=0, normalization_constant=39.43489815679818
    ):
        # sequence is a matrix of shape (num_beats, 12)
        # step 0: transpose node_chroma_matrix
        transposed_node_chroma_matrix = np.roll(node_chroma_matrix, transposition, axis=1)
        # step 1: binarize sequence
        sequence = (sequence > 0).astype(np.float32)
        # step 2: compute matrix of hamming distances between sequence and each node chroma vector
        hamming_distances = np.sum(
            (transposed_node_chroma_matrix[:, np.newaxis, :] != sequence[np.newaxis, :, :]), axis=2
        )
        # step 3: apply sigmoid to get scores
        hamming_scores = 1 / (1 + np.exp(hamming_distances))
        likelihood = hamming_scores / normalization_constant
        return np.log(likelihood).T

    def build_start_probs(self, mode="uniform"):
        num_beats = self.measures[-1].subnodes[-1].num + 1
        if mode == "uniform":
            start_probs = np.ones(num_beats) * (1 / num_beats)
        else:
            raise ValueError(f"Unknown start probability mode: {mode}")
        return start_probs

    def decode(self, sequence, transposition=0):
        self.build_subgraph()
        score, result = _hmmc.viterbi(
            self.build_start_probs(),
            self.build_transition_matrix(),
            self.compute_log_likelihood(self.build_node_chroma_matrix(), sequence, transposition),
        )
        return score, result


class LeadSheetConnector:
    def add_start_end_nodes(self, x, edge_index, edge_type, figures):
        start_node = len(x)
        end_node = len(x) + 1
        x.append(np.zeros_like(x[0]))
        x.append(np.zeros_like(x[0]))
        edge_index.append([start_node, 0])
        edge_type.append("start")
        edge_index.append([self.find_full_stop().num, end_node])
        edge_type.append("end")
        figures.append("START")
        figures.append("STOP")
        return x, edge_index, edge_type, figures

    def to_pyg(self):
        x = []
        edge_index = []
        edge_type = []
        figures = []
        for i, measure in enumerate(self.measures):
            assert i == measure.num
            x.append(measure.generate_chroma_matrix())
            # if measure.children is not None:
            if len(measure.children) > 0:
                for value, child in measure.children.items():
                    edge_index.append([measure.num, child.num])
                    edge_type.append(value)

            figure = []
            for beat_state in measure.beat_states:
                figure.append(beat_state.figure.replace(" ", ""))
            figures.append(" ".join(figure))
        x, edge_index, edge_type, figures = self.add_start_end_nodes(
            x, edge_index, edge_type, figures
        )
        return {
            "x": x,
            "edge_index": np.array(edge_index),
            "edge_type": edge_type,
            "title": self.score.metadata.title,
            "figure": figures,
        }

    @staticmethod
    def draw_one_graph(
        ax,
        edges,
        label=None,
        node_emb=None,
        layout=None,
        special_color=False,
        pos=None,
        figures=None,
    ):
        """draw a graph with networkx based on adjacency matrix (edges)
        graph labels could be displayed as a title for each graph
        node_emb could be displayed in colors
        """
        graph = nx.DiGraph()
        _edges = zip(edges[0], edges[1])
        _graph = nx.Graph()
        # print(list(edges))
        _graph.add_edges_from(_edges)
        _edges = zip(edges[0], edges[1])
        graph.add_edges_from(_edges)
        if layout == "custom":
            node_pos = pos
        elif layout == "tree":
            node_pos = nx.nx_agraph.graphviz_layout(_graph, prog="dot")
        else:
            node_pos = layout(_graph)
        # add colors according to node embeding
        if (node_emb is not None) or special_color:
            color_map = []
            node_list = [node[0] for node in graph.nodes(data=True)]
            for i, node in enumerate(node_list):
                # just ignore this branch
                if special_color:
                    if len(node_list) == 3:
                        crt_color = (1, 0, 0)
                    elif len(node_list) == 5:
                        crt_color = (0, 1, 0)
                    elif len(node_list) == 4:
                        crt_color = (1, 1, 0)
                    else:
                        special_list = (
                            [(1, 0, 0)] * 3 + [(0, 1, 0)] * 5 + [(1, 1, 0)] * 4
                        )
                        crt_color = special_list[i]
                else:
                    crt_node_emb = node_emb[node]
                    # map float number (node embeding) to a color
                    crt_color = cm.gist_rainbow(crt_node_emb, bytes=True)
                    crt_color = (
                        crt_color[0] / 255.0,
                        crt_color[1] / 255.0,
                        crt_color[2] / 255.0,
                        crt_color[3] / 255.0,
                    )
                color_map.append(crt_color)

            nx.draw_networkx_nodes(
                graph, node_pos, node_color=color_map, nodelist=node_list, ax=ax
            )
            nx.draw_networkx_edges(graph, node_pos, ax=ax)
            nx.draw_networkx_labels(graph, node_pos, ax=ax)
        else:
            nx.draw_networkx(graph, node_pos, ax=ax)
            for i, pos in node_pos.items():
                ax.text(
                    pos[0] + 0.03, pos[1], figures[i], fontsize=6
                )  # , ha='center', va='center')
            ax.set_title(label)

    def visualize(self, figsize=(10, 10), use_figures=True):
        plt.figure(figsize=figsize)
        pyg_sample = self.to_pyg()
        if use_figures:
            figures = pyg_sample["figure"]
        else:
            figures = ["" for _ in range(len(pyg_sample["figure"]))]
        self.draw_one_graph(
            plt.axes(),
            pyg_sample["edge_index"].T,
            layout=nx.layout.spring_layout,
            figures=figures,
            label=pyg_sample["title"],
        )


class State:
    """
    Class for monitoring the current state of the unrolling process
    Used for graph traversal
    """

    def __init__(self, transposition_plan=None, function_plan=None):
        # for planning
        # self.chorus_countdown = planned_repeats

        if transposition_plan is None:
            self.transposition_plan = [0, 0, 0, 0]
        else:
            self.transposition_plan = transposition_plan

        planned_repeats = len(self.transposition_plan)
        if function_plan is None:
            if planned_repeats == 1:
                self.function_plan = ["head"]
            elif planned_repeats == 2:
                self.function_plan = ["head", "head"]
            elif planned_repeats > 2:
                self.function_plan = (
                    ["head"] + ["solo"] * (planned_repeats - 2) + ["head"]
                )
        else:
            self.function_plan = function_plan

        # for monitoring:
        self.status = "START"
        self.section_title = None
        self.current_bar_number = None
        self.current_transposition = None

        # for navigation:
        self.reset()

        # for communication
        self.inbox = "back_to_top"

    def copy(self):
        new_state = State(
            transposition_plan=self.transposition_plan, function_plan=self.function_plan
        )
        new_state.status = self.status
        new_state.section_title = self.section_title
        new_state.current_bar_number = self.current_bar_number
        new_state.current_transposition = self.current_transposition
        new_state.inbox = self.inbox
        new_state.stop_at_fine = self.stop_at_fine
        new_state.jump_to_coda = self.jump_to_coda
        new_state.jump_to_segno = self.jump_to_segno
        new_state.current_repeat_num = self.current_repeat_num
        new_state.visited_repeat_starts = self.visited_repeat_starts.copy()
        return new_state

    def reset(self):
        self.current_repeat_num = 0
        self.visited_repeat_starts = set()
        self.inbox = None
        self.stop_at_fine = False
        self.jump_to_coda = False
        self.jump_to_segno = False

    def setup_last_chorus(self):
        pass
        # self.jump_to_coda = True

    def modify(self, measure):
        if self.inbox == "back_to_top":
            # self.chorus_countdown -= 1
            self.current_transposition = self.transposition_plan.pop(0)
            self.current_function = self.function_plan.pop(0)
            self.reset()
            if self.function_plan:
                self.setup_last_chorus()
        if measure == None:
            self.status = "END"
            self.current_bar_number = None
            return
        if measure.rehearsal_mark is not None:
            self.section_title = measure.rehearsal_mark
        self.current_bar_number = str(int(measure.num))
        if measure.repeat_start:
            if measure.num not in self.visited_repeat_starts:
                self.current_repeat_num = 0
                self.visited_repeat_starts.add(measure.num)
            self.current_repeat_num += 1
            self.status = f"REPEAT_{self.section_title}"
        if measure.repeat_end:
            pass
        if measure.ending is not None:
            self.status = f"ENDING_{self.current_repeat_num}"
        if measure.text_expression == "D.C. al 1st":
            self.status = "DC_AL_1ST_ENDING"
            self.stop_at_fine = True
            self.current_repeat_num = 0
        if measure.text_expression == "D.C. al 2nd":
            self.status = "DC_AL_2ND_ENDING"
            self.stop_at_fine = True
            self.current_repeat_num = 1
        if measure.text_expression == "D.C. al 3rd":
            self.status = "DC_AL_3RD_ENDING"
            self.stop_at_fine = True
            self.current_repeat_num = 2
        if measure.text_expression == "D.S. al 2nd":
            self.status = "DS_AL_2ND_ENDING"
            self.stop_at_fine = True
            self.current_repeat_num = 1
        if measure.DaCapoAlCoda:
            self.status = "DC_AL_CODA"
            self.jump_to_coda = True
            self.current_repeat_num = 0
        if measure.DalSegnoAlCoda:
            self.status = "DAL_SEGNO_AL_CODA"
            self.jump_to_coda = True
            self.current_repeat_num = 0
        if measure.DaCapoAlFine:
            self.status = "DC_AL_FINE"
            self.stop_at_fine = True
            self.current_repeat_num = 0

    def __repr__(self):
        return f"{self.status}, {self.section_title}, barnum {self.current_bar_number}, repnum {self.current_repeat_num}"


class BeatState:
    def __init__(self, chord, number):
        self.chord = chord
        self.figure = chord.figure
        self.chroma = chord.generate_chroma_features()
        self.chroma_matrix = chord.get_chroma_matrix()
        self.number = number


class Chord:
    def generate_chroma_features(self):
        pitches_chroma = [0 for _ in range(12)]
        bass_chroma = [0 for _ in range(12)]
        root_chroma = [0 for _ in range(12)]
        for pitch in self.pitches:
            pitches_chroma[pitch.midi % 12] = 1
        if self.bass is not None:
            bass_chroma[self.bass.midi % 12] = 1
        if self.root is not None:
            root_chroma[self.root.midi % 12] = 1
        return {
            "pitches": pitches_chroma,
            "bass": bass_chroma,
            "root": root_chroma,
        }

    def get_chroma_matrix(self):
        return np.array(
            [self.chroma["pitches"], self.chroma["bass"], self.chroma["root"]]
        ).T[np.newaxis, :, :]

    def __init__(self, chord=music21.harmony.NoChord()):
        self.chord = chord
        self.figure = chord.figure
        self.root = chord.root()
        self.bass = chord.bass()
        self.pitches = chord.pitches
        self.quality = chord.quality
        self.chordKind = chord.chordKind
        self.beat = chord.beat
        self.chroma = self.generate_chroma_features()


class Measure:
    def __init__(self, measure, endings=[], num=None):
        self.init_markers()
        self.measure = measure
        self.immediate_next = None
        self.children = {}
        self.num = num
        self.parse(endings)

    def init_markers(self):
        self.markers = {
            "solo_start": False,
            "solo_end": False,
            "head_start": False,
            "head_end": False,
            "intro_start": False,
            "intro_end": False,
            "verse_start": False,
            "verse_end": False,
        }

    def _parse_ending(self, endings):
        self.ending = None
        for k in endings:
            if k["bar_number"] == self.num:
                self.ending = k["ending_number"]

    def _parse_rehearsal_mark(self):
        self.rehearsal_mark = None
        for k in self.measure.getElementsByClass("RehearsalMark"):
            self._raw_rehearsal_mark = k
            self.rehearsal_mark = k.content

    def _parse_text_expression(self):
        self.text_expression = None
        for k in self.measure.getElementsByClass("TextExpression"):
            self.text_expression = k.content

    def _parse_DaCapoAlCoda(self):
        self.DaCapoAlCoda = False
        for k in self.measure.getElementsByClass("DaCapoAlCoda"):
            self.DaCapoAlCoda = True

    def _parse_DaCapoAlFine(self):
        self.DaCapoAlFine = False
        for k in self.measure.getElementsByClass("DaCapoAlFine"):
            self.DaCapoAlFine = True

    def _parse_DalSegnoAlCoda(self):
        self.DalSegnoAlCoda = False
        for k in self.measure.getElementsByClass("DalSegnoAlCoda"):
            self.DalSegnoAlCoda = True

    def _parse_repeat(self):
        self.repeat_start = False
        self.repeat_end = False
        self._raw_repeats = []
        for k in self.measure.getElementsByClass("Repeat"):
            self._raw_repeats.append(k)
            if k.direction == "start":
                self.repeat_start = True
            elif k.direction == "end":
                self.repeat_end = True

    def _parse_fine(self):
        self.fine = False
        for k in self.measure.getElementsByClass("Fine"):
            self.fine = True

    def _parse_coda(self):
        self.coda = False
        self.coda_beginning = False
        self.goto_coda = False
        self._coda_offset = None
        for k in self.measure.getElementsByClass("Coda"):
            self.coda = True
            self._coda_offset = k.offset
            if k.offset == 0:
                self.coda_beginning = True
            elif k.offset > 0:
                self.goto_coda = True

    def _parse_segno(self):
        self.segno = False
        for k in self.measure.getElementsByClass("Segno"):
            self.segno = True

    def _parse_chords(self):
        self.chords = []
        for k in self.measure.getElementsByClass("Chord"):
            self.chords.append(Chord(k))

    def _parse_time_signature(self):
        self.time_signature = None
        for k in self.measure.getElementsByClass("TimeSignature"):
            self.time_signature = {
                "ratio": k.ratioString,
                "numerator": k.numerator,
                "denominator": k.denominator,
            }

    def _parse_barlines(self):
        self.raw_barlines = []
        self.barline_left = None
        self.barline_right = None
        for k in self.measure.getElementsByClass("Barline"):
            self.raw_barlines.append(k)
            if k.location == "left":
                direction = None
                if "Repeat" in k.classes:
                    direction = k.direction
                self.barline_left = {
                    "repeat": "Repeat" in k.classes,
                    "direction": direction,
                    "type": k.type,
                    "location": k.location,
                }
            elif k.location == "right":
                direction = None
                if "Repeat" in k.classes:
                    direction = k.direction
                self.barline_right = {
                    "repeat": "Repeat" in k.classes,
                    "direction": direction,
                    "type": k.type,
                    "location": k.location,
                }
            else:
                raise ValueError(f"Unknown barline location {k.location}")

    def parse(self, endings):
        self._parse_rehearsal_mark()
        self._parse_text_expression()
        self._parse_repeat()
        self._parse_ending(endings)
        self._parse_fine()
        self._parse_coda()
        self._parse_segno()
        self._parse_DaCapoAlCoda()
        self._parse_DalSegnoAlCoda()
        self._parse_DaCapoAlFine()
        self._parse_chords()
        self._parse_time_signature()
        self._parse_barlines()

    def generate_chroma_matrix(self, transposition=0, channel="pitches") -> np.ndarray:
        """
        generates ndarray of dimensions (number_of_quarters, 12, 3)
        last dimension is for pitches, bass and root
        """
        if not hasattr(self, "beat_states"):
            self.generate_beat_states()
        matrix = np.concatenate([x.chroma_matrix for x in self.beat_states], axis=0)
        transposed_matrix = np.roll(matrix, transposition, axis=1)
        if channel == "pitches":
            return transposed_matrix[:, :, 0]
        elif channel == "bass":
            return transposed_matrix[:, :, 1]
        elif channel == "root":
            return transposed_matrix[:, :, 2]
        elif channel == "all":
            return transposed_matrix
        else:
            raise ValueError(f"Unknown channel {channel}")

    def generate_beat_states(self, start_number=0):
        number = start_number
        self.beat_states = []
        number_of_quarters = (
            self.time_signature["numerator"] / self.time_signature["denominator"]
        ) * 4
        assert number_of_quarters == int(number_of_quarters), "should be integer"
        number_of_quarters = int(number_of_quarters)
        current_beat = 0
        chord = Chord()
        chord_starts = [int(chord.beat) - 1 for chord in self.chords]
        chord_durations = [
            int(self.chords[i + 1].beat) - int(chord.beat)
            for i, chord in enumerate(self.chords[:-1])
        ]
        chord_durations.append(number_of_quarters - int(self.chords[-1].beat) + 1)
        for i, chord in enumerate(self.chords):
            # 0, 1
            while current_beat < chord_starts[i] + chord_durations[i]:
                self.beat_states.append(BeatState(chord, number))
                number += 1
                current_beat += 1
        while current_beat < number_of_quarters:
            self.beat_states.append(BeatState(chord, number))
            number += 1
            current_beat += 1
        self.beat_state_start_number = start_number
        self.beat_state_end_number = number

    def next(self, state):
        if self.fine and state.stop_at_fine:
            return None
        else:
            if "repeat_start" in self.children.keys() and state.current_repeat_num == 1:
                # NB test it for 3rd ending case
                return self.children["repeat_start"]
                # else:
                #    return self.children["next"]
            elif "ending_1" in self.children.keys():
                if state.current_repeat_num == 1:
                    return self.children["ending_1"]
                elif state.current_repeat_num == 2:
                    return self.children["ending_2"]
                elif state.current_repeat_num == 3:
                    return self.children["ending_3"]
            elif "D.C. al 1st" in self.children.keys():
                return self.children["D.C. al 1st"]
            elif "D.C. al 2nd" in self.children.keys():
                return self.children["D.C. al 2nd"]
            elif "D.C. al 3rd" in self.children.keys():
                return self.children["D.C. al 3rd"]
            elif "DaCapoAlCoda" in self.children.keys():
                return self.children["DaCapoAlCoda"]
            elif "D.S. al 2nd" in self.children.keys():
                return self.children["D.S. al 2nd"]
            elif "coda" in self.children.keys() and state.jump_to_coda:
                return self.children["coda"]
            elif "DalSegnoAlCoda" in self.children.keys():
                return self.children["DalSegnoAlCoda"]
            elif (
                "back_to_head_top" in self.children.keys()
                and state.function_plan
                and state.function_plan[0] == "head"
                and self.markers[f"{state.current_function}_end"]
            ):
                state.inbox = "back_to_top"
                return self.children["back_to_head_top"]
            elif (
                "back_to_solo_top" in self.children.keys()
                and state.function_plan
                and state.function_plan[0] == "solo"
                and self.markers[f"{state.current_function}_end"]
            ):
                state.inbox = "back_to_top"
                return self.children["back_to_solo_top"]
            else:
                return self.children.get("next", None)

    def __repr__(self):
        return f"Bar number {self.num}"

    def to_node(self):
        """
        generates numpy array (number_of_quarters x 12 semitones chroma x 3 channels - pitches, bass, root)
        """
        pitches = []
        bass = []
        root = []
        for state in self.beat_states:
            st_pitches = state.chroma["pitches"]
            st_bass = state.chroma["bass"]
            st_root = state.chroma["root"]
            pitches.append(st_pitches)
            bass.append(st_bass)
            root.append(st_root)
        pitches = np.array(pitches)
        bass = np.array(bass)
        root = np.array(root)
        return np.concatenate(
            [pitches[:, :, np.newaxis], bass[:, :, np.newaxis], root[:, :, np.newaxis]],
            axis=2,
        )


class LeadSheet(LeadSheetConnector, HMMConnectorMixin):
    def parse_spanners(self):
        endings = []
        for k in self.score.parts[0].getElementsByClass(music21.spanner.RepeatBracket):
            bar_number = k[0].number - 1
            ending_number = k.numberRange[0]
            assert len(k.numberRange) == 1, "should be this way afaik"
            endings.append({"bar_number": bar_number, "ending_number": ending_number})
        return endings

    def link_forward(self):
        for i in range(len(self.measures) - 1):
            self.measures[i].immediate_next = self.measures[i + 1]

    def link_backward(self):
        for i in range(len(self.measures) - 1):
            self.measures[i + 1].immediate_prev = self.measures[i]

    def _segment_repeats(self):
        segments = {}
        segment_counter = 0
        for bar in self.measures:
            if bar.repeat_start:
                segment_counter += 1
                segments[segment_counter] = []
                segments[segment_counter].append(bar)
            else:
                if segment_counter not in segments:
                    segments[segment_counter] = []
                segments[segment_counter].append(bar)
        self.repeat_segments = segments

    def accumulate_local_important_points(self, segment):
        important_points = {}
        for bar in segment:
            if bar.repeat_start:
                important_points["repeat_start"] = bar
            if bar.immediate_next is not None:
                if (bar.immediate_next.ending == 1) and (bar.ending is None):
                    important_points["forking_point"] = bar
            if bar.ending is not None:
                important_points[f"ending_{str(bar.ending)}"] = bar
            if bar.repeat_end:
                if "repeat_ends" not in important_points:
                    important_points["repeat_ends"] = []
                important_points["repeat_ends"].append(bar)
        return important_points

    def link_local_endings(self, segment):
        important_points = self.accumulate_local_important_points(segment)
        if "forking_point" in important_points.keys():
            for name, bar in important_points.items():
                if name.startswith("ending"):
                    important_points["forking_point"].children[name] = bar
        for repeat_end_bar in important_points.get("repeat_ends", []):
            repeat_end_bar.children["repeat_start"] = important_points["repeat_start"]

    def _fix_link_endings(self):
        if self.path.endswith("/Zingaro (Retrato Em Branco E Preto).musicxml"):
            self.measures[37].children["ending_2"] = self.measures[40]

    def link_endings(self):
        self._segment_repeats()
        for _, segment in self.repeat_segments.items():
            self.link_local_endings(segment)
        self._fix_link_endings()  # quick fix for Zingaro

    def propagate_endings(self):
        """
        Assuming that after an ending there is a new rehearsal mark
        """
        for i, bar in enumerate(self.measures[1:]):
            if bar.rehearsal_mark is None:
                if bar.ending is None:
                    bar.ending = self.measures[i].ending

    def propagate_time_signature(self):
        for i, bar in enumerate(self.measures[1:]):
            if bar.time_signature is None:
                bar.time_signature = self.measures[i].time_signature

    def link_coda(self):
        coda_instances = []
        for measure in self.measures:
            if measure.coda:
                coda_instances.append(measure)
        if len(coda_instances) == 1:
            # print("check coda!")
            # coda_instances[0].immediate_prev.children["coda"] = coda_instances[0]
            # coda_instances[0].immediate_prev.coda = True
            # wacky
            self.markers["solo_end"].children["coda"] = coda_instances[0]
            # self.markers['solo_end'].coda = True
        if len(coda_instances) == 2:
            coda_instances[0].children["coda"] = coda_instances[1]

    def link_jumps(self):
        for measure in self.measures:
            if measure.text_expression == "D.C. al 1st":
                measure.children["D.C. al 1st"] = self.measures[0]
            if measure.text_expression == "D.C. al 2nd":
                measure.children["D.C. al 2nd"] = self.measures[0]
            if measure.text_expression == "D.C. al 3rd":
                measure.children["D.C. al 3rd"] = self.measures[0]
            if measure.DaCapoAlCoda:
                measure.children["DaCapoAlCoda"] = self.measures[0]
            if measure.DaCapoAlFine:
                measure.children["DaCapoAlFine"] = self.measures[0]
            if measure.DalSegnoAlCoda:
                segno_instances = []
                for bar in self.measures:
                    if bar.segno:
                        segno_instances.append(bar)
                assert len(segno_instances) in [0, 1]
                if len(segno_instances) == 1:
                    measure.children["DalSegnoAlCoda"] = segno_instances[0]
            if measure.text_expression == "D.S. al 2nd":
                segno_instances = []
                for bar in self.measures:
                    if bar.segno:
                        segno_instances.append(bar)
                assert len(segno_instances) in [0, 1]
                if len(segno_instances) == 1:
                    measure.children["D.S. al 2nd"] = segno_instances[0]

    def _fix(self):
        self.measures[0].repeat_start = True
        if (
            self.path.endswith("/Diverse.musicxml")
            or self.path.endswith("/Segment.musicxml")
            or self.path.endswith("/Walkin' My Baby Back Home.musicxml")
        ):
            self.measures[8].ending = 2

    def generate_beat_states(self):
        number = 0
        for measure in self.measures:
            measure.generate_beat_states(number)
            number = measure.beat_state_end_number

    def parse_solo_info(self):
        """
        only markers found in Jazz 1410 playlist
        """
        self.solo_info = None
        SOLO_EXPLICIT_MARKERS = ["Solos", "(Solos)", "Solos:"]
        SOLO_INTERNAL_MARKERS = [
            "SOLOS A B C B",
            "solos on AABA",
            "Solos on AABA (alt. chanhes for head)",
            "solos on AB",
            "solos on AB (use alternate changes)",
        ]
        SOLO_BLUES_MARKERS = [
            "Solos on Bb Blues",
            "Solos on C- Blues",
            "Solos on F minor Blues",
            "solos on G Blues",
        ]
        SOLO_FREE_MARKERS = ["Solos free around Eb"]
        for measure in self.measures:
            if measure.text_expression in SOLO_EXPLICIT_MARKERS:
                measure.solo_start = True
                self.markers["solo_start"] = measure
                self.solo_info = f"solo starts at {measure.num}"
            if measure.text_expression in SOLO_INTERNAL_MARKERS:
                self.internal_solo = True
                self.solo_info = measure.text_expression
            if measure.text_expression in SOLO_BLUES_MARKERS:
                self.solo_info = measure.text_expression
            if measure.text_expression in SOLO_FREE_MARKERS:
                self.solo_info = measure.text_expression

    def build_external_solo_loop(self):
        SOLO_BLUES_MARKERS = {
            # "Solos on Bb Blues": "G_blues",
            "Solos on C- Blues": "Cminor_blues",
            "Solos on F minor Blues": "Fminor_blues",
            "solos on G Blues": "G_blues",
        }
        if self.solo_info is not None:
            if self.solo_info in SOLO_BLUES_MARKERS.keys():
                self.external_form_name = SOLO_BLUES_MARKERS[self.solo_info]
                solo_form = LeadSheet(
                    "/".join(self.path.split("/")[:-2])
                    + f"/external-solo-forms/{self.external_form_name}.musicxml"
                )
                solo_form.measures[0].repeat_start = True
                solo_form.measures[0].rehearsal_mark = "[EXT]Solo"
                solo_form.measures[-1].repeat_end = True
                total_measures = len(self.measures)
                for i, measure in enumerate(solo_form.measures):
                    solo_form.measures[i].num = total_measures + i
                    self.measures.append(solo_form.measures[i])
                self.markers["solo_start"] = self.measures[total_measures]
                self.markers["solo_end"] = self.measures[-1]
                self.markers["head_end"] = self.measures[total_measures - 1]

    def find_form_start(self):
        """
        get all measures with barline_lefts or rehearsal_mark
        """
        candidates = []
        for measure in self.measures:
            if measure.barline_left is not None or measure.rehearsal_mark is not None:
                candidates.append(measure)
        """
        version 2:
        return first measure with left barline that does not have "intro" or "verse" rehearsal mark
        """
        for candidate in candidates:
            if candidate.rehearsal_mark is None:
                return candidate
            else:
                if not candidate.rehearsal_mark in ["intro", "verse"]:
                    return candidate
        raise ValueError("No form start found")

    def find_solo_start(self):
        for measure in self.measures:
            if measure.solo_start:
                return measure
            return self.find_form_start()

    def find_form_end(self):
        candidates = []
        for measure in self.measures:
            if measure.rehearsal_mark == "[EXT]Solo":
                break
            if measure.barline_right is not None:
                candidates.append(measure)
            if measure.coda_beginning:
                candidates.append(
                    self.measures[measure.num - 1]
                )  # for song You Know I Care
                break  # ignore endpoints after coda started
        for candidate in candidates:
            if candidate.fine:
                return candidate
        if len(candidates) > 0:
            if candidates[-1].DaCapoAlCoda:
                return self.measures[-1]
            return candidates[-1]  # if no fine, return last barline before coda
        raise ValueError("No form end found")

    def find_full_stop(self):
        candidates = []
        for measure in self.measures:
            if measure.fine:
                candidates.append(measure)
        if len(candidates) == 0:
            if self.markers["head_end"] is None:
                return self.measures[-1]
            else:
                return self.markers["head_end"]
        return candidates[-1]

    def close_solo_loop(self):
        self.markers["solo_end"].children["back_to_solo_top"] = self.markers[
            "solo_start"
        ]
        self.markers["head_end"].children["back_to_head_top"] = self.markers[
            "head_start"
        ]
        self.markers["solo_end"].children["back_to_head_top"] = self.markers[
            "head_start"
        ]
        self.markers["head_end"].children["back_to_solo_top"] = self.markers[
            "solo_start"
        ]

    def init_markers(self):
        self.markers = {
            "solo_start": None,
            "solo_end": None,
            "head_start": None,
            "head_end": None,
            "intro_start": None,
            "intro_end": None,
            "verse_start": None,
            "verse_end": None,
        }
        self.external_form_name = None
        self.internal_solo = None

    def parse_markers(self):
        """
        problem: sometimes this info is not in rehearsal_mark, but in text_expression
        """
        intro_flag = False
        verse_flag = False
        solo_flag = False
        head_flag = False
        for i, measure in enumerate(self.measures):
            if (measure.rehearsal_mark is not None) or (
                measure.text_expression is not None
            ):

                if solo_flag:
                    self.markers["solo_end"] = self.measures[i - 1]
                    solo_flag = False
                if intro_flag:
                    self.markers["intro_end"] = self.measures[i - 1]
                    intro_flag = False
                if verse_flag:
                    self.markers["verse_end"] = self.measures[i - 1]
                    verse_flag = False

                if measure.rehearsal_mark in ["intro", "Intro", "INTRO"]:
                    self.markers["intro_start"] = measure
                    intro_flag = True
                elif measure.rehearsal_mark in ["verse", "Verse", "VERSE"]:
                    self.markers["verse_start"] = measure
                    verse_flag = True
                elif measure.text_expression in [
                    "[EXT]Solo",
                    "Solo",
                    "Solos",
                    "(Solos)",
                    "Solos x4",
                    "Solos:",
                ]:
                    self.markers["solo_start"] = measure
                    solo_flag = True
                    if head_flag:
                        self.markers["head_end"] = self.measures[i - 1]
                        head_flag = False
                elif measure.text_expression in ["Melody"]:
                    self.markers["head_start"] = measure
                    head_flag = True
                else:
                    if not head_flag and (measure.rehearsal_mark is not None):
                        self.markers["head_start"] = measure
                        head_flag = True

        if measure.text_expression in [None, "Melody"]:
            if head_flag:
                # self.markers["head_end"] = self.measures[-1]
                head_flag = False
            if solo_flag:
                self.markers["solo_end"] = self.measures[-1]
                solo_flag = False
            if intro_flag:
                self.markers["intro_end"] = self.measures[-1]
                intro_flag = False
            if verse_flag:
                self.markers["verse_end"] = self.measures[-1]
                verse_flag = False

    def fix_markers(self):
        if self.title == "Celia":
            self.markers["solo_start"] = self.markers["head_start"]
            self.markers["solo_end"] = self.measures[29]
        if self.title == "Bouncin' With Bud":
            self.markers["solo_start"] = self.markers["head_start"]
            self.markers["solo_end"] = self.measures[28]
        if self.title == "S.O.S.":
            self.markers["solo_start"] = self.markers["head_start"]
            self.markers["solo_end"] = self.measures[31]
        if self.title == "Hallucinations":
            self.markers["solo_start"] = self.markers["head_start"]
            self.markers["solo_end"] = self.measures[24]

    def propagate_markers(self):
        if self.markers["head_start"] is None:
            self.markers["head_start"] = self.measures[0]
        if self.markers["head_end"] is None:
            self.markers["head_end"] = self.find_full_stop()
        if self.markers["solo_start"] is None:
            self.markers["solo_start"] = self.markers["head_start"]
        if self.markers["solo_end"] is None:
            # self.markers["solo_end"] = self.markers["head_end"]
            self.markers["solo_end"] = self.find_form_end()

        # propagate leadsheet markers to measure markers
        if self.markers["intro_start"] is not None:
            self.markers["intro_start"].markers["intro_start"] = True
        if self.markers["intro_end"] is not None:
            self.markers["intro_end"].markers["intro_end"] = True
        if self.markers["verse_start"] is not None:
            self.markers["verse_start"].markers["verse_start"] = True
        if self.markers["verse_end"] is not None:
            self.markers["verse_end"].markers["verse_end"] = True
        if self.markers["solo_start"] is not None:
            self.markers["solo_start"].markers["solo_start"] = True
        if self.markers["solo_end"] is not None:
            self.markers["solo_end"].markers["solo_end"] = True
        if self.markers["head_start"] is not None:
            self.markers["head_start"].markers["head_start"] = True
        if self.markers["head_end"] is not None:
            self.markers["head_end"].markers["head_end"] = True

    def connect_successors(self):
        """
        each measure has an "immediate_next" field, which is a reference to the next measure
        but sometimes there is no transition between these measures, so here we are
        differentiating these cases
        """
        for measure in self.measures:
            if measure.immediate_next:
                jump_flag = any(
                    key in measure.children
                    for key in ["DalSegnoAlCoda", "DaCapoAlCoda", "DaCapoAlFine"]
                )
                if measure.text_expression in [
                    "D.C. al 1st",
                    "D.C. al 2nd",
                    "D.C. al 3rd",
                    "D.S. al 2nd",
                ]:
                    jump_flag = True

                # two endings in a row should never be connected
                if measure.immediate_next.ending:
                    if measure.ending != measure.immediate_next.ending:
                        jump_flag = True

                # if in current bar barline_right is final and next bar is coda
                if measure.barline_right:
                    if measure.immediate_next.coda:
                        jump_flag = True

                if not jump_flag:
                    measure.children["next"] = measure.immediate_next

    def __init__(self, path):
        self.path = path
        self.title = path.split("/")[-1].split(".musicxml")[0]
        self.score = music21.converter.parse(path)
        measures = self.score.parts[0].getElementsByClass("Measure")
        self.endings = self.parse_spanners()
        self.measures = [
            Measure(measure, self.endings, i) for i, measure in enumerate(measures)
        ]
        self.init_markers()
        self.parse_solo_info()
        self.build_external_solo_loop()
        self.parse_markers()
        self.fix_markers()
        self.propagate_markers()
        self._fix()
        self.link_forward()
        self.link_backward()
        self.link_endings()
        self.link_jumps()
        self.link_coda()
        self.propagate_endings()
        self.propagate_time_signature()
        self.generate_beat_states()
        self.close_solo_loop()
        self.connect_successors()

    @staticmethod
    def transform_beat_chords(sequence):
        return (
            "[START]_"
            + "_[BARLINE]_".join(["_".join(bar) for bar in sequence])
            + "_[END]"
        ).split("_")

    def unroll(self, transpose=None, max_length=np.inf):
        traversal = []
        state = State(transposition_plan=transpose)
        measure = self.measures[0]
        state.modify(measure)
        counter = max_length
        while state.status != "END" and counter > 0:
            traversal.append(
                {
                    "measure": measure,
                    "state": state.copy(),
                }
            )
            measure = measure.next(state)
            state.modify(measure)
            counter -= 1
        return traversal

    def straight(self):
        """same as unroll, but straight through, without repeats"""

        beat_states = []
        beat_chords = []

        for measure in self.measures:
            for beat_state in measure.beat_states:
                beat_states.append(beat_state)
            beat_chords.append(
                [beat_state.figure for beat_state in measure.beat_states]
            )

        return {
            "chroma": np.concatenate([x.chroma_matrix for x in beat_states], axis=0),
            "beat_chords": [state.figure for state in beat_states],
            "elements": self.transform_beat_chords(beat_chords),
            "labels": [state.number for state in beat_states],
        }

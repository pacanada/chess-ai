def get_en_passant_index_offset(en_passant_allowed: List[int])-> int:
    """Transform from en_passant encoding used in chess.state to simple 0 to 16 encoding. 0 nothing,
    1-8 white rank 3 and 9-16 black rank 5"""
    if len(en_passant_allowed)==0:
        return 0
    index = en_passant_allowed[0]
    if index >=16 and index <= 23:
        # a3 would be 1
        return index - 15 
    if index >=40 and index <= 47:
        return index - 31

def get_castling_encoding(c_e: List[int]):
    # {"Q": 0, "K": 1, "q": 2, "k": 3}
    bit_string = "".join(["1" if 0 in c_e else "0", "1" if 1 in c_e else "0", "1" if 2 in c_e else "0", "1" if 3 in c_e else "0"])
    #16 for [0,1,2,3] (1111)
    return int(bit_string, 2)

def encode_state(state: State):
    # TODO: this is far from efficient, there must be a clever way to encode a chess position for nn
    en_passant_encoding: int = get_en_passant_index_offset(state.en_passant_allowed)
    encoded_state = state.board+[state.turn]+[en_passant_encoding] + [get_castling_encoding(state.castling_rights)]
    return encoded_state
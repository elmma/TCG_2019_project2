#pragma once
#include <array>
#include <iostream>
#include <iomanip>

#include <cmath>

/**
 * array-based board for 2048
 *
 * index (1-d form):
 *  (0)  (1)  (2)  (3)
 *  (4)  (5)  (6)  (7)
 *  (8)  (9) (10) (11)
 * (12) (13) (14) (15)
 *
 */
class board {
public:
	typedef uint32_t cell;
	typedef std::array<cell, 4> row;
	typedef std::array<row, 4> grid;
	typedef uint64_t data;
	typedef int reward;	
	typedef unsigned op; //add for sliding rule

public:
	board() : tile(), attr(0), last_act(5), round(0) {}
	board(const grid& b, data v = 0) : tile(b), attr(v) {}
	board(const board& b) = default;
	board& operator =(const board& b) = default;
	

	operator grid&() { return tile; }
	operator const grid&() const { return tile; }
	row& operator [](unsigned i) { return tile[i]; }
	const row& operator [](unsigned i) const { return tile[i]; }
	cell& operator ()(unsigned i) { return tile[i / 4][i % 4]; }
	const cell& operator ()(unsigned i) const { return tile[i / 4][i % 4]; }

	data info() const { return attr; }
	data info(data dat) { data old = attr; attr = dat; return old; }

public:
	bool operator ==(const board& b) const { return tile == b.tile; }
	bool operator < (const board& b) const { return tile <  b.tile; }
	bool operator !=(const board& b) const { return !(*this == b); }
	bool operator > (const board& b) const { return b < *this; }
	bool operator <=(const board& b) const { return !(b < *this); }
	bool operator >=(const board& b) const { return !(*this < b); }

public:

	/**
	 * place a tile (index value) to the specific position (1-d form index)
	 * return 0 if the action is valid, or -1 if not
	 *
	 * (comment add) : we modify its valid symbols for tiles
	 * (1,2) -> (1,2,3)
	 */
	reward place(unsigned pos, cell tile) {
		if (pos >= 16) return -1;
		// add tile 3 here
		if (tile != 1 && tile != 2 && tile != 3) return -1;
		operator()(pos) = tile;
		return 0;
	}

	/**
	 * apply an action to the board
	 * return the reward of the action, or -1 if the action is illegal
	 * (add) we set last_act here for evil placing tile
	 */
	reward slide(unsigned opcode) {

		last_act = opcode & 0b11;	// set action record
		round++;
		//std::cout << "----------------------------------------------------" <<std::endl;
		//std::cout << round <<std::endl;
		//std::cout << "----------------------------------------------------" <<std::endl;
		switch (opcode & 0b11) {
		case 0: return slide_up();
		case 1: return slide_right();
		case 2: return slide_down();
		case 3: return slide_left();
		default: return -1;
		}
	}

	// my comment 
	/**  tile : 2d-array ,top : remaining-tile index
	 *   consider a row tiles : ( * for current hold tile , x for top index)  
	 *	 a 0 b 0 -> 0 a b 0 -> a 0 b 0 -> a 0 0 b -> a b 0 0 -> done
	 *   x*		 -> x *	   ->   x *   ->   x   * ->     x   
	 *   
	 *	 here we have to change the merge rule and the movement ,others already done 
	 *
	 * 	 we introduce new sliding here , we should note the rules(facts) : (use left)
	 *   1. for rows ,no double merge occur ,only the left pair merge
	 *   2. if first tile empty(for row) ,then no merge ,all tiles shift left 1
	 *   3. if a merge occur ,shift left 1 for later tiles
	 */
	

	// test slide_left for threes
	// for rows, holding itself if no move,holding zero if move 

	reward slide_left() {
		board prev = *this;
		reward score = 0;	// +1 if eliminate tiles
		for (int r = 0; r < 4; r++) {
			auto& row = tile[r];	// pick up row
			int hold = row[0];	// we hold the left at first
			for (int c = 1; c < 4; c++) {
				int tile = row[c];				
				//row[c] = 0;		// in here we do not reset tile
				if (hold) {
					// holding the same tile -> merge ->move to prev index
					if (tile > 2 && tile == hold) {		// 3n case
						row[c-1] = tile+hold;
						score += (tile+hold);	// ?
						hold = 0;
					} else if ((tile+hold==3) && tile<3) {	// 1+2 case
						row[c-1] = tile+hold;
						score += (tile+hold);	// ?
						hold = 0;
					} else {
					// not the same ,change hold to this
						hold = tile;
					}
				} else {
					row[c-1]=tile;	// if holding zero , then prev vacant -> move
					// hold = 0;  // already zero
				}
			}
			row[3] = hold;	// always put back
		}
		return (*this != prev) ? score : -1;
	}

	reward slide_right() {
		reflect_horizontal();
		reward score = slide_left();
		reflect_horizontal();
		return score;
	}
	reward slide_up() {
		rotate_right();
		reward score = slide_right();
		rotate_left();
		return score;
	}
	reward slide_down() {
		rotate_right();
		reward score = slide_left();
		rotate_left();
		return score;
	}

	void transpose() {
		for (int r = 0; r < 4; r++) {
			for (int c = r + 1; c < 4; c++) {
				std::swap(tile[r][c], tile[c][r]);
			}
		}
	}

	void reflect_horizontal() {
		for (int r = 0; r < 4; r++) {
			std::swap(tile[r][0], tile[r][3]);
			std::swap(tile[r][1], tile[r][2]);
		}
	}

	void reflect_vertical() {
		for (int c = 0; c < 4; c++) {
			std::swap(tile[0][c], tile[3][c]);
			std::swap(tile[1][c], tile[2][c]);
		}
	}

	/**
	 * rotate the board clockwise by given times
	 */
	void rotate(int r = 1) {
		switch (((r % 4) + 4) % 4) {
		default:
		case 0: break;
		case 1: rotate_right(); break;
		case 2: reverse(); break;
		case 3: rotate_left(); break;
		}
	}

	void rotate_right() { transpose(); reflect_horizontal(); } // clockwise
	void rotate_left() { transpose(); reflect_vertical(); } // counterclockwise
	void reverse() { reflect_horizontal(); reflect_vertical(); }

public:
	friend std::ostream& operator <<(std::ostream& out, const board& b) {
		out << "+------------------------+" << std::endl;
		for (auto& row : b.tile) {
			out << "|" << std::dec;
			for (auto t : row) out << std::setw(6) << ((1 << t) & -2u);
			out << "|" << std::endl;
		}
		out << "+------------------------+" << std::endl;
		return out;
	}

// add for passing action
public: 
	const op get_last_act() const {
		return this->last_act;
	}

// add some decode function for transform the space	
int dec(int input, bool mode = false) const{
		uint32_t space[] = {0, 1, 2, 3, 6, 12, 24, 48, 96, 192, 384, 768, 1536, 3072, 6144};
		if(mode) return space[input];
		return (input<5) ? input : static_cast<int>(log2(input/6)+4);
	}

private:
	grid tile;
	data attr;
	op last_act;	// add last_act for sliding rule
	int round;
};

/**
 * Proportional Veto Core (PVC) computation utilities
 */

// Simple assertion function for debugging
function assert(condition: boolean, message?: string): asserts condition {
	if (!condition) {
		throw new Error(message || 'Assertion failed');
	}
}

export function areNumbersApproximatelyEqual(num1: number, num2: number, tolerance: number = Number.EPSILON): boolean {
	return Math.abs(num1 - num2) < tolerance;
}

export type Alternative = string;
export type VoterPreference = Alternative[];
export type PreferenceProfile = VoterPreference[];

export interface VetoCoalitionResult {
	coalition: number[]; // indices of voters in the veto coalition
	selectedAlternative: Alternative;
	preferredAlternatives: Alternative[];
	dashboardValues: {
		T: number;
		T_size: number;
		v_T: number;
		B: Alternative[];
		lambda_B_over_P: number;
	};
}

/**
 * Compute the Proportional Veto Core (PVC) for a given preference profile by successive elimination
 * @param preferences - Matrix where preferences[i][j] is the alternative at rank i (0-indexed) for voter j
 * @param alternatives - List of all alternatives
 * @returns Array of alternatives in the PVC
 */
export function computePVC(preferences: string[][], alternatives: Alternative[]): Alternative[] {
	console.log('=== Starting computePVC ===');
	const m = alternatives.length;
	const n = preferences[0]?.length || 0; // number of voters (columns)
	console.log(`Number of alternatives (m): ${m}`);
	console.log(`Number of voters (n): ${n}`);
	console.log('Alternatives:', alternatives);
	console.log('Preferences matrix:', preferences);

	if (m === 0 || n === 0) {
		console.log('Early return: empty alternatives or no voters');
		return [];
	}

	// Map alternatives to indices for number representation
	const altToIndex = new Map<Alternative, number>();
	alternatives.forEach((alt, idx) => altToIndex.set(alt, idx));
	console.log('Alternative to index mapping:', altToIndex);

	// Convert preference matrix to profile format (each voter's complete ordering)
	// Note that `profile` is [voter][alternative] while `preferences` is the transpose
	console.log('Converting preference matrix to profile format...');
	const profile: number[][] = [];
	for (let voter = 0; voter < n; voter++) {
		const voterPrefs: number[] = [];
		for (let rank = 0; rank < m; rank++) {
			const alt = preferences[rank][voter];
			const index = altToIndex.get(alt) ?? -1;
			assert(index != -1);
			voterPrefs.push(index);
		}
		console.log(`Voter ${voter} preferences (as indices):`, voterPrefs);
		profile.push(voterPrefs);
	}
	console.log('Complete profile (numeric):', profile);

	// veto by consumption
	// init each alternative tank
	const tanks = [];
	for (let i = 0; i < m; ++i) {
		tanks.push(1.0);
	}
	const remainingAlts = new Set(Array.from({ length: m }, (_, i) => i));
	// run the clock
	const eps = 1e-9;
	while (remainingAlts.size > 1) {
		console.log(`remainingAlts`,remainingAlts);
		// init count voter in each alternative
		const num_voter_eating: number[] = Array(m).fill(0.0);
		for (let voter = 0; voter < n; ++voter) {
			const voterProfile = profile[voter];
			++num_voter_eating[voterProfile[voterProfile.length - 1]];
		}
		console.log(`num_voter_eating`,num_voter_eating);
		// find t_delta
		let t_delta = 1.0;
		for (let alt = 0; alt < m; ++alt) {
			if (tanks[alt] == 0 || num_voter_eating[alt] == 0) continue;
			t_delta = Math.min(t_delta, tanks[alt] / num_voter_eating[alt]);
		}
		console.log("t_delta",t_delta);
		// let t_delta pass
		const eliminatedNow = [];
		for (let alt = 0; alt < m; ++alt) {
			if (tanks[alt] == 0) continue;
			tanks[alt] -= t_delta * num_voter_eating[alt];
			if (tanks[alt] < eps) {
				console.log("killed", alt);
				tanks[alt] = 0;
				remainingAlts.delete(alt);
				eliminatedNow.push(alt);
			}
		}
		console.log('tanks',tanks)
		console.log('eliminatedNow',eliminatedNow);
		// remove alts from voters rankings
		for (let voter = 0; voter < n; ++voter) {
			while (profile[voter].length > 0 && tanks[profile[voter][profile[voter].length - 1]] == 0) {
				profile[voter].pop();
			}
		}
		if (remainingAlts.size == 0) {
			// ties
			return numbersToAlphabets(eliminatedNow);
		}
	}

	return numbersToAlphabets([...remainingAlts]);
}

function numbersToAlphabets(numbers: number[]): string[] {
	return numbers.map(numberToAlphabet);
}


function numberToAlphabet(num: number): string {
	// Supports 0 -> 'a', 1 -> 'b', ..., 25 -> 'z', 26 -> 'aa', etc.
	let result = '';
	num = Math.floor(num);
	do {
		result = String.fromCharCode(97 + (num % 26)) + result;
		num = Math.floor(num / 26) - 1;
	} while (num >= 0);
	return result;
}


/**
 * Check if a coalition of voters can veto an alternative
 * @param alternative - The alternative to check
 * @param coalition - Indices of voters in the coalition
 * @param preferences - Preference matrix
 * @param alternatives - List of all alternatives
 * @returns Object with veto result and preferred alternatives
 */
function checkVetoCoalition(
	alternative: Alternative,
	coalition: number[],
	preferences: string[][],
	alternatives: Alternative[]
): { canVeto: boolean; preferredAlternatives: Alternative[] } {
	console.log(`trying to veto ${alternative} with coalition, `, coalition);
	if (coalition.length === 0) {
		return { canVeto: false, preferredAlternatives: [] };
	}

	// Convert preference matrix to profile format for coalition voters
	const coalitionProfiles: number[][] = [];
	const altToIndex = new Map<Alternative, number>();
	alternatives.forEach((alt, idx) => altToIndex.set(alt, idx));

	const m = alternatives.length;
	for (const voterIndex of coalition) {
		const voterPrefs: number[] = [];
		for (let rank = 0; rank < m; rank++) {
			const alt = preferences[rank][voterIndex];
			const index = altToIndex.get(alt) ?? -1;
			assert(index != -1);
			voterPrefs.push(index);
		}
		coalitionProfiles.push(voterPrefs);
	}

	const targetAltIndex = altToIndex.get(alternative);
	if (targetAltIndex === undefined) {
		return { canVeto: false, preferredAlternatives: [] };
	}

	// Find alternatives preferred by all coalition members over the target alternative
	const preferredByAll: Set<number> = new Set();

	// Start with alternatives preferred by first coalition member
	const firstVoterPrefs = coalitionProfiles[0];
	const targetRankInFirst = firstVoterPrefs.indexOf(targetAltIndex);
	if (targetRankInFirst === -1) {
		return { canVeto: false, preferredAlternatives: [] };
	}

	for (let rank = 0; rank < targetRankInFirst; rank++) {
		preferredByAll.add(firstVoterPrefs[rank]);
	}

	// Intersect with preferences of other coalition members
	for (let i = 1; i < coalitionProfiles.length; i++) {
		const voterPrefs = coalitionProfiles[i];
		const targetRankInVoter = voterPrefs.indexOf(targetAltIndex);
		if (targetRankInVoter === -1) {
			return { canVeto: false, preferredAlternatives: [] };
		}

		const voterPreferred = new Set<number>();
		for (let rank = 0; rank < targetRankInVoter; rank++) {
			voterPreferred.add(voterPrefs[rank]);
		}

		// Keep only alternatives preferred by both current voter and previous intersection
		const newPreferredByAll = new Set<number>();
		for (const alt of preferredByAll) {
			if (voterPreferred.has(alt)) {
				newPreferredByAll.add(alt);
			}
		}
		preferredByAll.clear();
		newPreferredByAll.forEach(alt => preferredByAll.add(alt));
	}

	// A coalition can veto if they satisfy the PVC veto condition:
	// |T|*(m-1)/n >= 1 - |B|/m, where T is coalition size, B is preferred alternatives
	const n = preferences[0]?.length || 0;
	const T_size = coalition.length;
	const B_size = preferredByAll.size;

	const veto_power = Math.ceil((T_size * m) / n) - 1;
	const veto_size = m - B_size;
	const canVeto = veto_power >= veto_size;
	const preferredAlternatives = Array.from(preferredByAll).map(idx => alternatives[idx]);
	console.log(`T_size / n: ${T_size / n}`);
	console.log(`1-B_size / m: ${1 - B_size / m}`);
	console.log(`canVeto: ${canVeto}`);
	console.log(`preferredAlternatives:`, preferredAlternatives);

	return { canVeto, preferredAlternatives };
}

/**
 * Compute a veto coalition for a given alternative not in the PVC
 * @param alternative - The alternative to find a veto coalition for
 * @param preferences - Matrix where preferences[i][j] is the i-th ranked alternative for voter j
 * @param alternatives - List of all alternatives
 * @param pvc - Current PVC
 * @returns Veto coalition result with coalition members and dashboard values
 */
export function computeVetoCoalition(
	alternative: Alternative,
	preferences: string[][],
	alternatives: Alternative[],
	pvc: Alternative[]
): VetoCoalitionResult {
	const m = alternatives.length;
	const n = preferences[0]?.length || 0; // number of voters

	// Iterate over all 2^n possible coalitions (excluding empty set)
	let bestCoalition: number[] = [];
	let bestPreferred: Alternative[] = [];

	for (let coalitionMask = 1; coalitionMask < (1 << n); coalitionMask++) {
		// Convert bit mask to coalition indices
		const coalition: number[] = [];
		for (let voter = 0; voter < n; voter++) {
			if (coalitionMask & (1 << voter)) {
				coalition.push(voter);
			}
		}

		const result = checkVetoCoalition(alternative, coalition, preferences, alternatives);

		if (result.canVeto && result.preferredAlternatives.length > 0) {
			bestCoalition = coalition;
			bestPreferred = result.preferredAlternatives;
		}
	}

	// Dashboard values
	const dashboardValues = {
		T: bestCoalition.length,
		T_size: bestCoalition.length,
		v_T: Math.ceil(bestCoalition.length * m / n) - 1,
		B: bestPreferred,
		lambda_B_over_P: bestPreferred.length / m
	};

	return {
		coalition: bestCoalition,
		selectedAlternative: alternative,
		preferredAlternatives: bestPreferred,
		dashboardValues
	};
}

/**
 * Validate a voter's preference ordering
 * @param voterPrefs - Single voter's preference ordering
 * @param alternatives - List of all valid alternatives
 * @returns true if the preference ordering is valid
 */
export function validateVoterPreferences(voterPrefs: Alternative[], alternatives: Alternative[]): boolean {
	// Check for correct length
	if (voterPrefs.length !== alternatives.length) {
		return false;
	}

	// Check for duplicates
	const unique = new Set(voterPrefs);
	if (unique.size !== voterPrefs.length) {
		return false;
	}

	// Check for empty entries
	if (voterPrefs.some(pref => !pref)) {
		return false;
	}

	// Check if all entries are valid alternatives
	return voterPrefs.every(pref => alternatives.includes(pref));
}

/**
 * Convert preference matrix format to preference profile format
 * @param preferences - Matrix where preferences[i][j] is the i-th ranked alternative for voter j
 * @returns Array where each element is a voter's complete preference ordering
 */
export function matrixToProfile(preferences: string[][]): PreferenceProfile {
	const m = preferences.length; // number of alternatives
	const n = preferences[0]?.length || 0; // number of voters

	const profile: PreferenceProfile = [];

	for (let voter = 0; voter < n; voter++) {
		const voterPrefs: Alternative[] = [];
		for (let rank = 0; rank < m; rank++) {
			voterPrefs.push(preferences[rank][voter]);
		}
		profile.push(voterPrefs);
	}

	return profile;
}

/**
 * Generate alternatives a, b, c, ... based on the number m
 * @param m - Number of alternatives
 * @returns Array of alternative labels
 */
export function generateAlternatives(m: number): Alternative[] {
	return Array.from({ length: m }, (_, i) => String.fromCharCode(97 + i));
}
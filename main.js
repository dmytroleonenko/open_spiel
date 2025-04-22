// main.js

// --- Log Script Start ---
console.log("main.js: Script loaded.");

// --- Constants and Globals ---
const BOARD_WIDTH = 700;
const BOARD_HEIGHT = 450;
const POINT_WIDTH = BOARD_WIDTH / 14;
const POINT_HEIGHT = BOARD_HEIGHT * 0.42;
const BAR_WIDTH = POINT_WIDTH;
const CHECKER_RADIUS = POINT_WIDTH * 0.4;
const CHECKER_DIAMETER = CHECKER_RADIUS * 2;
const CHECKER_STACK_OFFSET = CHECKER_DIAMETER * 0.6;
const MAX_VISIBLE_STACK = Math.floor(POINT_HEIGHT / CHECKER_STACK_OFFSET) + 1;
const WHITE_COLOR = '#FFFFFF';
const BLACK_COLOR = '#5C3D2E';
const STROKE_COLOR = '#000000';
const POINT_COLOR_A = '#C19A6B';
const POINT_COLOR_B = '#A0784F';
const BOARD_BG_COLOR = '#e0cda9';
const BAR_COLOR = '#8B4513';

let testData = {};
console.log("main.js: Constants and globals defined.");

// --- Canvas Drawing Functions (Keep implementations as before) ---
function getPointCoordinates(pointIndex) {
    const visualPoint = pointIndex + 1;
    let x, y_base, isTop;
    if (visualPoint >= 1 && visualPoint <= 6) {
        x = BOARD_WIDTH - (visualPoint * POINT_WIDTH) - (BAR_WIDTH / 2); y_base = BOARD_HEIGHT - CHECKER_RADIUS; isTop = false;
    } else if (visualPoint >= 7 && visualPoint <= 12) {
        x = BOARD_WIDTH - (visualPoint * POINT_WIDTH) - (BAR_WIDTH / 2) - BAR_WIDTH; y_base = BOARD_HEIGHT - CHECKER_RADIUS; isTop = false;
    } else if (visualPoint >= 13 && visualPoint <= 18) {
        x = POINT_WIDTH * (visualPoint - 13) + (BAR_WIDTH / 2); y_base = CHECKER_RADIUS; isTop = true;
    } else { // 19-24
        x = POINT_WIDTH * (visualPoint - 13) + BAR_WIDTH + (BAR_WIDTH / 2); y_base = CHECKER_RADIUS; isTop = true;
    }
    return { x: x + POINT_WIDTH / 2, y_base, isTop, visualPoint };
}

function drawBoard(ctx) {
    if (!ctx) { console.error("drawBoard: ctx is null"); return; }
    ctx.clearRect(0, 0, BOARD_WIDTH, BOARD_HEIGHT);
    ctx.fillStyle = BOARD_BG_COLOR; ctx.fillRect(0, 0, BOARD_WIDTH, BOARD_HEIGHT);
    for (let i = 0; i < 24; i++) {
        const { x, isTop, visualPoint } = getPointCoordinates(i);
        const color = ((visualPoint % 2) === (isTop ? 0 : 1)) ? POINT_COLOR_A : POINT_COLOR_B;
        ctx.fillStyle = color; ctx.beginPath();
        if (isTop) { ctx.moveTo(x - POINT_WIDTH / 2, 0); ctx.lineTo(x + POINT_WIDTH / 2, 0); ctx.lineTo(x, POINT_HEIGHT); }
        else { ctx.moveTo(x - POINT_WIDTH / 2, BOARD_HEIGHT); ctx.lineTo(x + POINT_WIDTH / 2, BOARD_HEIGHT); ctx.lineTo(x, BOARD_HEIGHT - POINT_HEIGHT); }
        ctx.closePath(); ctx.fill();
        ctx.fillStyle = '#555'; ctx.font = '10px sans-serif'; ctx.textAlign = 'center';
        if (isTop) ctx.fillText(visualPoint, x, 12); else ctx.fillText(visualPoint, x, BOARD_HEIGHT - 5);
    }
    ctx.fillStyle = BAR_COLOR; ctx.fillRect(BOARD_WIDTH / 2 - BAR_WIDTH / 2, 0, BAR_WIDTH, BOARD_HEIGHT);
    ctx.strokeStyle = '#5C3D2E'; ctx.lineWidth = 5; ctx.strokeRect(2.5, 2.5, BOARD_WIDTH - 5, BOARD_HEIGHT - 5);
}

function drawChecker(ctx, x, y, player) {
    if (!ctx) { console.error("drawChecker: ctx is null"); return; }
    ctx.beginPath(); ctx.arc(x, y, CHECKER_RADIUS, 0, Math.PI * 2);
    ctx.fillStyle = (player === 0) ? WHITE_COLOR : BLACK_COLOR; ctx.fill();
    ctx.strokeStyle = STROKE_COLOR; ctx.lineWidth = 1; ctx.stroke();
}

function drawPointCheckers(ctx, pointIndex, player, count) {
    if (!ctx) { console.error("drawPointCheckers: ctx is null"); return; }
    if (count <= 0) return;
    const { x, y_base, isTop } = getPointCoordinates(pointIndex);
    let currentOffset = CHECKER_STACK_OFFSET;
    if (count > MAX_VISIBLE_STACK) {
        currentOffset = (POINT_HEIGHT - CHECKER_DIAMETER) / (count - 1);
        currentOffset = Math.max(currentOffset, CHECKER_RADIUS * 0.2);
    }
    for (let i = 0; i < count; i++) {
        const y = isTop ? (y_base + i * currentOffset) : (y_base - i * currentOffset);
        if (y < -CHECKER_RADIUS || y > BOARD_HEIGHT + CHECKER_RADIUS) break;
        drawChecker(ctx, x, y, player);
    }
    if (count > MAX_VISIBLE_STACK) {
        ctx.fillStyle = (player === 0) ? BLACK_COLOR : WHITE_COLOR; ctx.font = 'bold 12px sans-serif'; ctx.textAlign = 'center';
        const lastVisibleY = isTop ? (y_base + (MAX_VISIBLE_STACK - 1) * currentOffset) : (y_base - (MAX_VISIBLE_STACK - 1) * currentOffset);
        const textY = isTop ? lastVisibleY + CHECKER_RADIUS * 0.5 : lastVisibleY - CHECKER_RADIUS * 0.5;
        try { ctx.fillText(count, x, textY); } catch (e) { console.error("Error drawing checker count text", e); }
    }
}

function renderTestState(canvasId, initialState) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) { return; }
    if (!initialState) { console.warn("Initial state missing for canvas ID:", canvasId); return; }
    const ctx = canvas.getContext('2d');
    if (!ctx) { console.error("Could not get 2D context for canvas:", canvasId); return; }
    try { // Wrap rendering in try-catch
        drawBoard(ctx);
        if (initialState.board && initialState.board.length === 2 && initialState.board[0]?.length === 24 && initialState.board[1]?.length === 24) {
            for (let player = 0; player < 2; player++) {
                for (let pos = 0; pos < 24; pos++) {
                    const count = Number(initialState.board[player][pos]);
                    if (!isNaN(count) && count > 0) {
                        drawPointCheckers(ctx, pos, player, count);
                    }
                }
            }
        } else {
            console.error("Invalid board state format for canvas:", canvasId, JSON.stringify(initialState.board));
            ctx.fillStyle = 'red'; ctx.font = '16px sans-serif'; ctx.textAlign = 'center';
            ctx.fillText('Invalid Board Data', BOARD_WIDTH / 2, BOARD_HEIGHT / 2);
        }
    } catch (e) {
        console.error(`Error during renderTestState for ${canvasId}:`, e);
        ctx.fillStyle = 'red'; ctx.font = '16px sans-serif'; ctx.textAlign = 'center';
        ctx.fillText('Render Error', BOARD_WIDTH / 2, BOARD_HEIGHT / 2);
    }
}

// --- LocalStorage & Verification Logic ---
async function calculateChecksum(string) {
    const encoder = new TextEncoder();
    const data = encoder.encode(string);
    const hashBuffer = await crypto.subtle.digest('SHA-256', data);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    const hashHex = hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
    return hashHex;
}

async function saveSourceCode(filename, code) {
    console.log(`main.js: saveSourceCode called for ${filename}`);
    try {
        const currentChecksum = await calculateChecksum(code);
        let sourceFiles = JSON.parse(localStorage.getItem('sourceFiles') || '{}');
        let verificationStatuses = JSON.parse(localStorage.getItem('verificationStatuses') || '{}');
        const previousChecksum = sourceFiles[filename]?.checksum;
        let statusesInvalidated = false;

        sourceFiles[filename] = { code: code, checksum: currentChecksum };
        localStorage.setItem('sourceFiles', JSON.stringify(sourceFiles));

        if (previousChecksum && previousChecksum !== currentChecksum) {
            console.log(`main.js: Source code for ${filename} changed. Invalidating verifications.`);
            statusesInvalidated = true;
            for (const testId in verificationStatuses) {
                const status = verificationStatuses[testId];
                const checkbox = document.querySelector(`.verification-checkbox[data-testid="${testId}"][data-filename="${filename}"]`);
                if (checkbox && status.verified) {
                    status.verified = false;
                    status.verifiedChecksum = null;
                }
            }
        }

        if (statusesInvalidated) {
            localStorage.setItem('verificationStatuses', JSON.stringify(verificationStatuses));
            initializeVerificationStates(filename);
        } else if (!previousChecksum) {
            console.log(`main.js: Saved initial source code for ${filename}.`);
            initializeVerificationStates(filename);
        } else {
            console.log(`main.js: Resaved source code for ${filename} (checksum unchanged).`);
            // Maybe still re-init to be safe if user clicks save unnecessarily
            initializeVerificationStates(filename);
        }

    } catch (error) {
        console.error(`main.js: Error saving source code for ${filename}:`, error);
        if (error.name === 'QuotaExceededError') { alert('Error: LocalStorage quota exceeded.'); }
        else { alert(`An unexpected error occurred while saving source code for ${filename}. Check console.`); }
    }
}

function getSourceCode(filename) {
    // console.log(`main.js: getSourceCode called for ${filename}`);
    try {
        const sourceFiles = JSON.parse(localStorage.getItem('sourceFiles') || '{}');
        return sourceFiles[filename]; // Returns object or undefined
    } catch (error) {
        console.error(`main.js: Error retrieving source code for ${filename}:`, error);
        return null;
    }
}

function saveVerificationStatus(testId, filename, isVerified) {
    console.log(`main.js: saveVerificationStatus called for ${testId}, ${filename}, verified: ${isVerified}`);
    try {
        let verificationStatuses = JSON.parse(localStorage.getItem('verificationStatuses') || '{}');
        const sourceInfo = getSourceCode(filename);
        const currentChecksum = sourceInfo ? sourceInfo.checksum : null;

        if (!currentChecksum && isVerified) {
            console.warn(`main.js: Cannot verify test ${testId} for ${filename} because source code is not loaded.`);
            alert(`Please load the source code for ${filename} before verifying tests.`);
            const checkbox = document.querySelector(`.verification-checkbox[data-testid="${testId}"]`);
            if (checkbox) checkbox.checked = false;
            return;
        }

        verificationStatuses[testId] = {
            verified: isVerified,
            verifiedChecksum: isVerified ? currentChecksum : null
        };
        localStorage.setItem('verificationStatuses', JSON.stringify(verificationStatuses));
        console.log(`main.js: Saved verification status for ${testId}: ${isVerified}`);
        updateSuiteHighlight(filename);
    } catch (error) {
        console.error(`main.js: Error saving verification status for ${testId}:`, error);
        alert(`An error occurred while saving verification status for ${testId}. Check console.`);
    }
}

function getVerificationStatus(testId, filename) {
    // console.log(`main.js: getVerificationStatus called for ${testId}, ${filename}`);
    try {
        const verificationStatuses = JSON.parse(localStorage.getItem('verificationStatuses') || '{}');
        const status = verificationStatuses[testId];

        if (!status || !status.verified) { return false; }

        const currentSourceInfo = getSourceCode(filename);
        if (!currentSourceInfo) {
            console.warn(`main.js: Source code for ${filename} missing during verification check for ${testId}.`);
            return false;
        }
        if (status.verifiedChecksum !== currentSourceInfo.checksum) {
            console.warn(`main.js: Checksum mismatch for ${testId} in ${filename}.`);
            return false;
        }
        return true; // Verified and checksum matches
    } catch (error) {
        console.error(`main.js: Error in getVerificationStatus for ${testId} (${filename}):`, error);
        return false;
    }
}

function updateSuiteHighlight(filename) {
    console.log(`main.js: updateSuiteHighlight called for ${filename}`);
    const suiteElement = document.querySelector(`.test-suite[data-filename="${filename}"]`);
    if (!suiteElement) { console.warn(`main.js: Suite element not found for ${filename}`); return; }
    const checkboxes = suiteElement.querySelectorAll('.verification-checkbox');
    if (checkboxes.length === 0) { suiteElement.classList.remove('verified-suite'); return; }

    let allEffectivelyVerified = true;
    for (const checkbox of checkboxes) {
        if (!getVerificationStatus(checkbox.dataset.testid, filename)) {
            allEffectivelyVerified = false; break;
        }
    }
    if (allEffectivelyVerified) { suiteElement.classList.add('verified-suite'); }
    else { suiteElement.classList.remove('verified-suite'); }
    console.log(`main.js: Suite ${filename} highlight updated. All verified: ${allEffectivelyVerified}`);
}


// --- Code Snippet Display Logic ---
function findCodeBetweenAnchors(code, anchorName, isFunction = true) {
    if (!code || !anchorName) { console.error("findCodeBetweenAnchors: Missing code or anchorName"); return { startLine: 1, endLine: 1, snippet: code || "", highlightStart: 1, highlightEnd: 1 }; }
    console.log(`[findCodeBetweenAnchors] Called with anchorName: ${anchorName}, isFunction: ${isFunction}`); // DEBUG
    const lines = code.split('\n');
    const isTestId = anchorName.startsWith('test-');
    let prefix, suffix;
    if (isTestId) { prefix = `//StartTest: ${anchorName}`; suffix = `//EndTest: ${anchorName}`; }
    else { prefix = isFunction ? `//StartFunction: ${anchorName}` : `//StartTest: ${anchorName}`; suffix = isFunction ? `//EndFunction: ${anchorName}` : `//EndTest: ${anchorName}`; }
    console.log(`[findCodeBetweenAnchors] Searching for prefix: "${prefix}", suffix: "${suffix}"`); // DEBUG
    let startLine = -1, endLine = -1, endAnchorLine = -1;
    for (let i = 0; i < lines.length; i++) { if (lines[i].includes(prefix)) { startLine = i + 1; break; } }
    if (startLine === -1) { for (let i = 0; i < lines.length; i++) { if ((lines[i].includes('//StartTest:') || lines[i].includes('//StartFunction:')) && lines[i].includes(anchorName)) { startLine = i + 1; break; } } }
    if (startLine !== -1) {
        for (let i = startLine; i < lines.length; i++) { if (lines[i].includes(suffix)) { endAnchorLine = i; break; } }
        if (endAnchorLine === -1) { for (let i = startLine; i < lines.length; i++) { if ((lines[i].includes('//EndTest:') || lines[i].includes('//EndFunction:')) && lines[i].includes(anchorName)) { endAnchorLine = i; break; } } }
        if (endAnchorLine !== -1) { endLine = endAnchorLine + 1; } else { endLine = startLine; } // Default to highlighting only start if end missing
    }
    if (startLine === -1) { console.warn(`Anchors (Start) not found for ${anchorName}`); startLine = 1; endLine = 1; }
    console.log(`[findCodeBetweenAnchors] Found lines: startLine=${startLine}, endLine=${endLine}`); // DEBUG
    return { startLine, endLine, snippet: code, highlightStart: startLine, highlightEnd: endLine > 0 ? endLine : startLine };
}


function displaySourceInputModal(filename) {
    console.log(`main.js: displaySourceInputModal called for ${filename}`);
    const modal = $('#sourceInputModal');
    const sourceInfo = getSourceCode(filename);
    modal.find('#sourceFilename, #sourceFilenameRepeat').text(filename);
    modal.find('#targetFilenameInput').val(filename);
    modal.find('#sourceCodeInput').val(sourceInfo ? sourceInfo.code : '');
    modal.modal('show');
}


// --- HTML Generation Functions ---
function createTestCaseElement(caseData, filename) {
    // Destructure caseData, but handle initialState separately for safety
    const {
        id,
        title = "Untitled Test Case",
        objective = "",
        boardSetup = "",
        diceRoll = null,
        expectations = "",
        fileFunctionRef = "",
        rulesInvolved = "",
        notes = ""
        // DO NOT destructure initialState here with defaults for sub-properties
    } = caseData;

    // Explicitly handle initialState being null or undefined
    const initialState = caseData.initialState; // Get the value directly

    // Provide defaults if initialState or its properties are missing
    const state = {
        board: initialState?.board || null, // Use optional chaining ?.
        scores: initialState?.scores || [0, 0],
        playerToMove: initialState?.playerToMove !== undefined ? initialState.playerToMove : 0, // Check for undefined explicitly
        dice: initialState?.dice || null
    };

    const playerText = state.playerToMove === 0 ? 'White (X)' : 'Black (O)';
    const diceText = state.dice ? state.dice.join(', ') : 'N/A';
    const scoreX = state.scores[0];
    const scoreO = state.scores[1];

    // Generate unique IDs for spans within this test case
    const scoreXId = `score-x-${id}`;
    const scoreOId = `score-o-${id}`;
    const playerId = `player-${id}`;
    const diceId = `dice-${id}`;
    const canvasId = `board-canvas-${id}`;

    // Use template literals and conditional rendering
    // Check specifically for state.board before attempting to render canvas
    const canvasHtml = state.board
        ? `<canvas id="${canvasId}" width="${BOARD_WIDTH}" height="${BOARD_HEIGHT}"></canvas>`
        : '<p><em>No board visualization provided for this test.</em></p>';

    const html = `
        <div class="test-case" id="${id}">
            <h4>${title}</h4>
            <input type="checkbox" class="verification-checkbox" data-testid="${id}" data-filename="${filename}">
            <label style="font-weight: normal; display: inline;">Verify</label>
            ${objective ? `<p><strong>Objective:</strong> ${objective}</p>` : ''}
            ${boardSetup ? `<p><strong>Board Setup:</strong> ${boardSetup}</p>` : ''}
            ${diceRoll ? `<p><strong>Dice Roll:</strong> ${diceRoll}</p>` : ''}
            ${notes ? `<p><em>Note: ${notes}</em></p>` : ''}
            <div class="board-info">
                White (X) Score: <span id="${scoreXId}">${scoreX}</span> |
                Black (O) Score: <span id="${scoreOId}">${scoreO}</span> |
                To Move: <span id="${playerId}">${playerText}</span> |
                Dice: <span id="${diceId}">${diceText}</span>
            </div>
            ${canvasHtml}
            ${expectations ? `<p><strong>Expectations:</strong> ${expectations}</p>` : ''}
            ${fileFunctionRef ? `<p class="file-line-ref"><strong>File/Function Ref:</strong> <code>${fileFunctionRef}</code></p>` : ''}
            ${rulesInvolved ? `<p><strong>Rules Involved:</strong> ${rulesInvolved}</p>` : ''}
        </div>
    `;
    return html;
}

function createTestFunctionElement(functionData, filename) {
    const testCases = functionData.testCases || [];
    let html = `<div class="test-function"><h3 data-filename="${filename}" data-lines="${functionData.lines || 'N/A'}" title="Click to view source code">Function: \`${functionData.name || 'Unnamed'}\` ${functionData.lines ? `(lines ${functionData.lines})` : ''}</h3>`;
    if (testCases.length > 0) { testCases.forEach(caseData => { try { html += createTestCaseElement(caseData, filename); } catch (e) { console.error(`Error creating HTML for test case ${caseData?.id} in ${functionData?.name}:`, e); html += `<div class="alert alert-warning">Error rendering test case ${caseData?.id || '(unknown ID)'}.</div>`; } }); }
    else { html += `<p><em>No test cases defined.</em></p>`; }
    html += `</div>`; return html;
}

function createTestSuiteElement(suiteData) {
    if (!suiteData || !suiteData.filename) { console.error("Invalid suiteData:", suiteData); return '<div class="alert alert-danger">Error: Invalid test suite data.</div>'; }
    const filename = suiteData.filename; const safeFilename = filename.replace(/[^a-zA-Z0-9-_]/g, '-');
    const suiteId = `suite-${safeFilename}`; const contentId = `content-${safeFilename}`; const functions = suiteData.functions || [];
    let html = `<div class="test-suite" data-filename="${filename}"><h2 id="${suiteId}" class="suite-header" data-toggle="collapse" data-target="#${contentId}" aria-expanded="false" aria-controls="${contentId}"><span class="collapse-icon">[+]</span> Test Suite: <code>${filename}</code> <button class="btn btn-xs btn-default load-source-btn">Load Source</button></h2><div class="suite-content collapse" id="${contentId}">`;
    if (suiteData.error) { html += `<div class="alert alert-danger">Failed to load test data for this suite. ${suiteData.errorMessage || ''}</div>`; }
    else if (functions.length > 0) { functions.forEach(functionData => { try { html += createTestFunctionElement(functionData, filename); } catch (e) { console.error(`Error creating HTML for function ${functionData?.name} in ${filename}:`, e); html += `<div class="alert alert-warning">Error rendering function ${functionData?.name || '(unknown name)'}.</div>`; } }); }
    else { html += `<p><em>No functions defined.</em></p>`; }
    html += `</div></div>`; return html;
}


// --- Initialization and Event Handling ---
// Use an immediately-invoked function expression (IIFE) for the main setup
(async function () {
    console.log("main.js: IIFE starting.");

    // --- Log Start of IIFE ---
    console.log("main.js: Inside IIFE, starting main execution.");

    const testContainer = $('#test-container');
    if (!testContainer.length) {
        console.error("main.js: #test-container element not found! Cannot proceed.");
        // Attempt to add a message directly to body if container missing
        $('body').prepend('<div class="alert alert-danger" style="margin:20px;">Initialization Error: Critical element #test-container is missing in the HTML.</div>');
        return; // Stop execution if container is missing
    }
    console.log("main.js: Test container found.");

    const jsonFiles = [
        'long_narde_test_actions.json',
        'long_narde_test_bridges.json',
        'long_narde_test_endgame.json',
        'long_narde_test_basic.json',
        'long_narde_test_movement.json',
        'random_sim_test.json'
    ];
    console.log("main.js: JSON files list:", jsonFiles);

    // --- Load JSON Data ---
    try {
        console.log("main.js: Starting to fetch JSON data...");
        const fetchPromises = jsonFiles.map(file => {
            console.log(`main.js: Initiating fetch for ${file}`);
            return fetch(file)
                .then(response => {
                    console.log(`main.js: Received response for ${file}, status: ${response.status}`);
                    if (!response.ok) {
                        console.error(`main.js: HTTP error! status: ${response.status} for ${file}`);
                        return { filename: file.replace('.json', '.cc'), error: true, status: response.status, message: `HTTP ${response.status}` };
                    }
                    return response.json()
                        .then(data => {
                            // console.log(`main.js: Successfully parsed JSON for ${file}`);
                            return { ...data, filename: file.replace('.json', '.cc'), error: false }; // Ensure filename is correct
                        })
                        .catch(parseError => {
                            console.error(`main.js: Failed to parse JSON for ${file}:`, parseError);
                            return { filename: file.replace('.json', '.cc'), error: true, message: `JSON Parse Error: ${parseError.message}` };
                        });
                })
                .catch(networkError => {
                    console.error(`main.js: Network error fetching ${file}:`, networkError);
                    return { filename: file.replace('.json', '.cc'), error: true, message: `Network Error: ${networkError.message}` };
                });
        });

        console.log("main.js: Waiting for all fetch promises to settle...");
        const loadedSuites = await Promise.all(fetchPromises);
        console.log("main.js: All fetch promises settled.");

        let encounteredLoadError = false;
        testData = {}; // Reset global test data
        loadedSuites.forEach(suite => {
            if (suite && suite.filename) {
                testData[suite.filename] = suite;
                if (suite.error) {
                    console.error(`main.js: Error recorded for ${suite.filename}:`, suite.message || `HTTP Status ${suite.status}`);
                    encounteredLoadError = true;
                }
            } else {
                console.error("main.js: Loaded suite data is invalid (missing filename?):", suite);
                encounteredLoadError = true;
            }
        });

        if (encounteredLoadError) {
            console.warn("main.js: One or more JSON files failed to load or parse.");
        } else {
            console.log("main.js: All JSON data loaded and processed successfully.");
        }

        // --- Generate HTML ---
        console.log("main.js: Generating HTML content...");
        testContainer.empty(); // Clear loading indicator
        const filenamesInOrder = jsonFiles.map(f => f.replace('.json', '.cc'));

        filenamesInOrder.forEach(filename => {
            const suite = testData[filename];
            if (suite) {
                try { testContainer.append(createTestSuiteElement(suite)); }
                catch (e) { console.error(`main.js: Error generating HTML for suite ${filename}:`, e); testContainer.append(`<div class="alert alert-danger">Failed to render suite ${filename}. Check console.</div>`); }
            } else {
                // This case means the file load failed completely earlier
                testContainer.append(createTestSuiteElement({ filename: filename, error: true, errorMessage: 'Failed to load data' }));
            }
        });
        console.log("main.js: HTML generation complete.");

        // --- Render Canvases ---
        console.log("main.js: Rendering canvases...");
        let canvasRenderCount = 0;
        $('.test-case canvas').each(function () {
            const canvas = $(this); const canvasId = canvas.attr('id');
            const testCaseId = canvas.closest('.test-case').attr('id');
            const filename = canvas.closest('.test-suite').data('filename');
            if (!testCaseId || !filename) { console.error("Canvas missing testCaseId or filename:", canvasId); return; }

            let initialState = null; const suite = testData[filename];
            if (suite && !suite.error && suite.functions) {
                for (const func of suite.functions) {
                    if (func.testCases) { const testCase = func.testCases.find(tc => tc.id === testCaseId); if (testCase && testCase.initialState) { initialState = testCase.initialState; break; } }
                }
            }
            if (initialState) { renderTestState(canvasId, initialState); canvasRenderCount++; }
            else { /* Warn only if board was expected? */ }
        });
        console.log(`main.js: Canvas rendering attempted. ${canvasRenderCount} canvases potentially rendered.`);


        // --- Initialize Checkboxes and Suite Highlights ---
        console.log("main.js: Initializing verification states...");
        initializeVerificationStates();
        console.log("main.js: Verification states initialized.");

        // --- Apply Clickable Styling ---
        styleFileLineRefs();
        console.log("main.js: Clickable styling applied.");

        // --- Attach Event Handlers ---
        // IMPORTANT: Attach handlers *after* the content exists
        attachEventHandlers();
        console.log("main.js: Event handlers attached.");


    } catch (error) {
        // Catch errors during the async setup process itself
        console.error("main.js: CRITICAL ERROR during async setup:", error);
        testContainer.empty(); // Clear loading indicator if it's still there
        testContainer.html('<div class="alert alert-danger" style="margin:20px;">An unexpected critical error occurred during page setup. Please check the console.</div>');
    }

})(); // Execute the IIFE


// --- Helper Functions for Initialization and Styling ---
function initializeVerificationStates(targetFilename = null) {
    console.log(`main.js: initializeVerificationStates called ${targetFilename ? 'for ' + targetFilename : '(for all)'}`);
    const uniqueFilenames = new Set();
    let checkedCount = 0;
    let disabledCount = 0;
    $('.verification-checkbox').each(function () {
        const checkbox = $(this); const testId = checkbox.data('testid'); const filename = checkbox.data('filename');
        if (targetFilename && filename !== targetFilename) { return; }
        uniqueFilenames.add(filename);
        const effectivelyVerified = getVerificationStatus(testId, filename);
        const sourceInfo = getSourceCode(filename);
        checkbox.prop('checked', effectivelyVerified); if (effectivelyVerified) checkedCount++;
        checkbox.prop('disabled', !sourceInfo); if (!sourceInfo) disabledCount++;
        const label = checkbox.next('label');
        if (!sourceInfo) { if (label.length) label.attr('title', `Load source for ${filename} to verify.`); checkbox.closest('.test-case').addClass('source-missing'); }
        else { if (label.length) label.removeAttr('title'); checkbox.closest('.test-case').removeClass('source-missing'); }
    });
    console.log(`main.js: Verification state init: ${checkedCount} checked, ${disabledCount} disabled.`);
    const filenamesToUpdate = targetFilename ? [targetFilename] : Array.from(uniqueFilenames);
    filenamesToUpdate.forEach(filename => { if (filename) updateSuiteHighlight(filename); });
}

function styleFileLineRefs() {
    $('p:contains("File/Function Ref:")').each(function () { const $p = $(this); if (!$p.hasClass('file-line-ref')) { $p.addClass('file-line-ref').css('cursor', 'pointer').attr('title', 'Click to view source'); } });
    $('h3[data-filename][data-lines]').css('cursor', 'pointer').attr('title', 'Click to view source');
}

// --- Code Snippet Display Logic ---
// Helper function to calculate scroll position
function calculateScrollPosition(modal, targetLine) {
    const lineNumbers = modal.find('.hljs-ln-numbers');
    let targetElement = null;
    const modalBody = modal.find('.modal-body');

    if (lineNumbers.length > 0) {
        for (let i = 0; i < lineNumbers.length; i++) {
            const lineNum = parseInt(lineNumbers[i].getAttribute('data-line-number'));
            if (lineNum === targetLine) {
                targetElement = lineNumbers[i];
                break;
            }
        }
    }

    let scrollPosition = 0;
    if (targetElement && modalBody.length) {
        try {
             // position().top is relative to the offset parent (modal-body usually)
             const elemTop = $(targetElement).position().top;
             // scrollTop() is the current amount scrolled within modal-body
             const containerScrollTop = modalBody.scrollTop() || 0;
             // Target position: element's top relative to container + current scroll - offset
             // Ensure we don't scroll past the top
             scrollPosition = Math.max(0, elemTop + containerScrollTop - 50); // Adjust 50px offset as needed
        } catch (e) {
            console.error("Error calculating element position:", e);
            scrollPosition = 0; // Fallback to top
        }
    }
    // console.log(`Calculated scroll position for line ${targetLine}: ${scrollPosition}`); // Debug
    return scrollPosition;
}


function displayCodeSnippet(filename, code, anchorName, isFunction = true) {
    console.log(`main.js: displayCodeSnippet called: filename=${filename}, anchorName=${anchorName}, isFunction=${isFunction}`);

    const { startLine, endLine, snippet, highlightStart, highlightEnd } = findCodeBetweenAnchors(code, anchorName, isFunction);
    const modal = $('#codeViewModal');
    const codeElement = document.getElementById('codeSnippetDisplay');
    const preElement = codeElement ? codeElement.parentElement : null;

    if (!modal.length || !codeElement || !preElement) { console.error("Modal elements not found!"); return; }

    const entityType = isFunction ? 'Function' : 'Test';
    let titleText = `Source: ${filename} (${entityType}: ${anchorName})`;
    const hasHighlight = highlightStart > 0 && highlightEnd > 0 && highlightEnd >= highlightStart && (highlightStart > 1 || highlightEnd > 1);
    if (hasHighlight) { titleText += ` - Highlighted Lines ${highlightStart}-${highlightEnd}`; }
    else { titleText += ` (Anchors not found or invalid, showing full file)`; }

    modal.find('.modal-title').text(titleText);
    codeElement.textContent = snippet;

    // Clear previous highlight.js artifacts before reapplying
    $(preElement).find('*').removeClass('hljs');
    $(preElement).removeClass((i, c) => (c.match(/(^|\s)hljs\S+/g) || []).join(' '));
    codeElement.className = 'language-cpp';
    $(preElement).find('.hljs-ln').remove(); // Remove old line numbers

    try {
        hljs.highlightElement(codeElement);
        hljs.lineNumbersBlock(codeElement, { startFrom: 1 });
    } catch (e) {
        console.error("Error during syntax highlighting:", e);
    }

    // Store target line for scrolling (use 1 if no specific highlight)
    const targetScrollLine = hasHighlight ? highlightStart : 1;
    modal.data('targetScrollLine', targetScrollLine); // Store the line number itself

    modal.off('shown.bs.modal').one('shown.bs.modal', function () {
        const currentModal = $(this); // Use $(this) inside handler
        const modalBody = currentModal.find('.modal-body');
        const lineNumbers = currentModal.find('.hljs-ln-numbers');
        const codeLines = currentModal.find('.hljs-ln-code');

        // Clear any lingering inline background styles from previous views
        modalBody.find('.hljs-ln-numbers[style*="background-color"], .hljs-ln-code[style*="background-color"]').css('background-color', '');

        // Apply highlights and scroll *after* a short delay
        setTimeout(() => {
            console.log("[displayCodeSnippet setTimeout] Starting highlight logic."); // DEBUG
            let calculatedScrollPos = 0;
            const hasHighlight = highlightStart > 0 && highlightEnd > 0 && highlightEnd >= highlightStart && (highlightStart > 1 || highlightEnd > 1); // Recalculate here for clarity

            if (lineNumbers.length > 0 && codeLines.length > 0 && lineNumbers.length === codeLines.length) {
                // Apply background highlights
                if (hasHighlight) {
                    console.log(`[displayCodeSnippet setTimeout] Applying highlight from ${highlightStart} to ${highlightEnd}`); // DEBUG
                    let highlightsApplied = 0; // DEBUG
                    for (let i = 0; i < lineNumbers.length; i++) {
                        const lineNum = parseInt(lineNumbers[i].getAttribute('data-line-number'));
                        if (lineNum >= highlightStart && lineNum <= highlightEnd) {
                            // console.log(`[displayCodeSnippet setTimeout] Highlighting line ${lineNum}`); // DEBUG (can be noisy)
                            lineNumbers[i].style.backgroundColor = '#ffffc0';
                            codeLines[i].style.backgroundColor = '#ffffc0';
                            highlightsApplied++; // DEBUG
                        }
                    }
                    console.log(`[displayCodeSnippet setTimeout] Applied highlights to ${highlightsApplied} lines.`); // DEBUG
                } else {
                    console.log(`[displayCodeSnippet setTimeout] No valid highlight range found (start=${highlightStart}, end=${highlightEnd}). Skipping highlight application.`); // DEBUG
                }

                // Calculate scroll position *now*
                const lineToScrollTo = currentModal.data('targetScrollLine') || 1;
                calculatedScrollPos = calculateScrollPosition(currentModal, lineToScrollTo);

            } else {
                console.warn("[displayCodeSnippet setTimeout] Line number/code line elements mismatch during highlight/scroll attempt.");
                calculatedScrollPos = calculateScrollPosition(currentModal, 1);
            }

            // Store the freshly calculated position for the button
            currentModal.data('scrollPosition', calculatedScrollPos);
             console.log(`Stored final scroll position: ${calculatedScrollPos}`); // Debug

            // Perform the scroll
             console.log(`Attempting to scroll modal body to: ${calculatedScrollPos}`); // Debug
            if (modalBody.length) {
                 modalBody.scrollTop(calculatedScrollPos);
            } else {
                console.error("Modal body not found for scrolling!");
            }

        }, 150); // Delay helps ensure layout is stable
    });

    // Setup the button click handler (uses the stored scrollPosition)
    $('#scroll-to-highlight-btn').off('click').on('click', function () {
        const scrollPosition = modal.data('scrollPosition'); // Retrieve stored position
        console.log("Manual scroll button clicked, scrolling to:", scrollPosition); // Debug
        if (scrollPosition !== undefined && modal.find('.modal-body').length) {
            modal.find('.modal-body').animate({ scrollTop: scrollPosition }, 300);
        } else {
            console.warn("Scroll position not available or modal body missing for button click.");
        }
    });

    // Show the modal *after* all event handlers are set up
    modal.modal('show');
    modal.find('.modal-dialog').addClass('modal-lg'); // Ensure large modal
}


// --- Separate function for line number based display (if needed, or merge logic) ---
function displayCodeSnippetWithLines(filename, code, startLine, endLine) {
    console.log(`main.js: displayCodeSnippetWithLines called: ${filename} lines ${startLine}-${endLine}`);
    const modal = $('#codeViewModal');
    const codeElement = document.getElementById('codeSnippetDisplay');
    const preElement = codeElement ? codeElement.parentElement : null;

    if (!modal.length || !codeElement || !preElement) { console.error("Modal elements not found!"); return; }

    const highlightStart = Math.max(1, startLine);
    const highlightEnd = Math.max(highlightStart, endLine);
    // const hasHighlight = true; // Assume we always want to highlight for this function // This was already here implicitly

    modal.find('.modal-title').text(`Source: ${filename} (Lines: ${highlightStart}-${highlightEnd})`);
    codeElement.textContent = code;

    // Reset highlight/line numbers
    $(preElement).find('*').removeClass('hljs');
    $(preElement).removeClass((i, c) => (c.match(/(^|\s)hljs\S+/g) || []).join(' '));
    codeElement.className = 'language-cpp';
    $(preElement).find('.hljs-ln').remove();

    try {
        hljs.highlightElement(codeElement);
        hljs.lineNumbersBlock(codeElement, { startFrom: 1 });
    } catch (e) { console.error("Error during highlighting:", e); }

    // Store target line for scrolling
    const targetScrollLine = highlightStart;
    modal.data('targetScrollLine', targetScrollLine);

    modal.off('shown.bs.modal').one('shown.bs.modal', function () {
         const currentModal = $(this); // Use $(this) inside handler
         const modalBody = currentModal.find('.modal-body');
         const lineNumbers = currentModal.find('.hljs-ln-numbers');
         const codeLines = currentModal.find('.hljs-ln-code');

         // Clear any lingering inline background styles
         modalBody.find('.hljs-ln-numbers[style*="background-color"], .hljs-ln-code[style*="background-color"]').css('background-color', '');

        // Apply highlights and scroll *after* a short delay
        setTimeout(() => {
            console.log("[displayCodeSnippetWithLines setTimeout] Starting highlight logic."); // DEBUG
            let calculatedScrollPos = 0;

            if (lineNumbers.length > 0 && codeLines.length > 0 && lineNumbers.length === codeLines.length) {
                // Apply background highlights
                 console.log(`[displayCodeSnippetWithLines setTimeout] Applying highlight from ${highlightStart} to ${highlightEnd}`); // DEBUG
                 let highlightsApplied = 0; // DEBUG
                 for (let i = 0; i < lineNumbers.length; i++) {
                     const lineNum = parseInt(lineNumbers[i].getAttribute('data-line-number'));
                     if (lineNum >= highlightStart && lineNum <= highlightEnd) {
                         // console.log(`[displayCodeSnippetWithLines setTimeout] Highlighting line ${lineNum}`); // DEBUG (can be noisy)
                         lineNumbers[i].style.backgroundColor = '#ffffc0';
                         codeLines[i].style.backgroundColor = '#ffffc0';
                         highlightsApplied++; // DEBUG
                     }
                 }
                 console.log(`[displayCodeSnippetWithLines setTimeout] Applied highlights to ${highlightsApplied} lines.`); // DEBUG

                // Calculate scroll position *now*
                const lineToScrollTo = currentModal.data('targetScrollLine') || 1;
                calculatedScrollPos = calculateScrollPosition(currentModal, lineToScrollTo);

            } else {
                console.warn("[displayCodeSnippetWithLines setTimeout] Line number/code line elements mismatch.");
                calculatedScrollPos = calculateScrollPosition(currentModal, 1);
            }

            // Store the freshly calculated position for the button
            currentModal.data('scrollPosition', calculatedScrollPos);
             console.log(`Stored final scroll position (lines version): ${calculatedScrollPos}`); // Debug

            // Perform the scroll
             console.log(`Attempting to scroll modal body to: ${calculatedScrollPos} (lines version)`); // Debug
             if (modalBody.length) {
                modalBody.scrollTop(calculatedScrollPos);
             } else {
                 console.error("Modal body not found for scrolling (lines version)!");
             }

        }, 150); // Delay
    });

     // Setup the button click handler (uses the stored scrollPosition)
     $('#scroll-to-highlight-btn').off('click').on('click', function () {
        const scrollPosition = modal.data('scrollPosition'); // Retrieve stored position
        console.log("Manual scroll button clicked, scrolling to:", scrollPosition); // Debug
        if (scrollPosition !== undefined && modal.find('.modal-body').length) {
            modal.find('.modal-body').animate({ scrollTop: scrollPosition }, 300);
        } else {
            console.warn("Scroll position not available or modal body missing for button click.");
        }
    });

    // Show the modal *after* all event handlers are set up
    modal.modal('show');
    modal.find('.modal-dialog').addClass('modal-lg');
}


// --- Event Handlers Attachment Function ---
function attachEventHandlers() {
    console.log("main.js: Attaching event handlers...");
    // Use document body for reliably attaching handlers that need to work on dynamic content
    const eventTarget = $(document.body);

    // Collapse toggles (Delegate from body to dynamic elements)
    eventTarget.on('show.bs.collapse', '.test-suite .collapse', function () { $(this).closest('.test-suite').find('.collapse-icon').text('[-]'); });
    eventTarget.on('hide.bs.collapse', '.test-suite .collapse', function () { $(this).closest('.test-suite').find('.collapse-icon').text('[+]'); });
    eventTarget.on('shown.bs.collapse', '.test-suite .collapse', function () { styleFileLineRefs(); }); // Re-style when shown

    // Verification Checkbox
    eventTarget.on('change', '.verification-checkbox', function () {
        const checkbox = $(this); const testId = checkbox.data('testid'); const filename = checkbox.data('filename'); const isVerified = checkbox.prop('checked');
        saveVerificationStatus(testId, filename, isVerified);
        if (isVerified) { setTimeout(() => $('#jump-to-unverified-btn').trigger('click'), 100); } // Use trigger
    });

    // Function Header Click
    eventTarget.on('click', '.test-function h3', function () {
        console.log("main.js: Function header clicked.");
        const $h3 = $(this); const filename = $h3.data('filename'); const lines = $h3.data('lines'); const sourceInfo = getSourceCode(filename);
        if (!filename) { console.error("Missing data-filename on h3"); return; }
        if (sourceInfo && sourceInfo.code) {
            const headerText = $h3.text(); const functionMatch = headerText.match(/Function:\s*`?([\w:]+)`?/i);
            if (functionMatch && functionMatch[1]) { displayCodeSnippet(filename, sourceInfo.code, functionMatch[1], true); }
            else if (lines && lines !== 'N/A') { const [start, end] = lines.split('-').map(Number); if (!isNaN(start) && !isNaN(end)) { displayCodeSnippetWithLines(filename, sourceInfo.code, start, end); } else { console.error(`Invalid lines: ${lines}`); displayCodeSnippet(filename, sourceInfo.code, filename, false); } }
            else { console.error(`Cannot extract function name or lines from: ${headerText}`); displayCodeSnippet(filename, sourceInfo.code, filename, false); }
        } else { console.warn(`Source code not found for ${filename}.`); alert(`Please load the source code for ${filename} first.`); }
    });

    // Load Source Button
    eventTarget.on('click', '.load-source-btn', function (event) {
        console.log("main.js: Load Source button clicked.");
        event.stopPropagation(); const filename = $(this).closest('.test-suite').data('filename');
        if (filename) { displaySourceInputModal(filename); } else { console.error("Could not find filename for Load Source button."); }
    });

    // File/Function Reference Click
    eventTarget.on('click', 'p.file-line-ref', function () {
        console.log("main.js: File/Ref link clicked.");
        const $p = $(this); const $testCase = $p.closest('.test-case'); const $suite = $testCase.closest('.test-suite');
        const filename = $suite.data('filename'); const testId = $testCase.attr('id');
        if (!filename || !testId) { console.error("Missing filename or testId for ref click"); return; }
        const sourceInfo = getSourceCode(filename);
        if (sourceInfo && sourceInfo.code) { displayCodeSnippet(filename, sourceInfo.code, testId, false); } // Assume ref always points to test case
        else { console.warn(`Source missing for ${filename}`); alert(`Please load the source code for ${filename} first.`); }
    });

    // --- Global Event Handlers (attached to static elements outside the container) ---

    // Modal Save Button
    $('#saveSourceCodeBtn').off('click').on('click', async function () {
        console.log("main.js: Save Source button clicked.");
        const filename = $('#targetFilenameInput').val(); const code = $('#sourceCodeInput').val();
        if (filename && code) { await saveSourceCode(filename, code); $('#sourceInputModal').modal('hide'); }
        else { alert("Error: Filename or code missing."); }
    });

    // Jump Button
    $('#jump-to-unverified-btn').off('click').on('click', function () {
        console.log("main.js: Jump button clicked.");
        let foundUnverified = false;
        $('.verification-checkbox').each(function () { // Iterate through existing checkboxes
            const checkbox = $(this); const testId = checkbox.data('testid'); const filename = checkbox.data('filename');
            if (!getVerificationStatus(testId, filename)) {
                console.log(`main.js: Jumping to ${testId}`);
                foundUnverified = true; const testCaseElement = checkbox.closest('.test-case'); const suiteContentElement = checkbox.closest('.suite-content');
                if (testCaseElement.length > 0 && suiteContentElement.length > 0) {
                    const performScrollAndHighlight = () => { testCaseElement[0].scrollIntoView({ behavior: 'smooth', block: 'center' }); testCaseElement.css({ 'background-color': '#fff8dc', 'transition': 'background-color 0.3s ease' }); setTimeout(() => { const isChecked = checkbox.prop('checked'); const originalBg = isChecked ? '#f5fff5' : '#f9f9f9'; testCaseElement.css('background-color', originalBg); }, 1500); };
                    if (!suiteContentElement.hasClass('in')) { $('.suite-content.collapse.in').not(suiteContentElement).collapse('hide'); suiteContentElement.one('shown.bs.collapse', performScrollAndHighlight); suiteContentElement.collapse('show'); }
                    else { performScrollAndHighlight(); }
                } else { console.error(`Cannot find elements for jump target ${testId}`); }
                return false; // Stop .each
            }
        });
        if (!foundUnverified) { alert("All tests appear to be verified!"); }
    });
    console.log("main.js: All event handlers attached.");
}
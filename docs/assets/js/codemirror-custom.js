function initCodeMirrorForCodeTags() {
    const codeTags = document.querySelectorAll('code.language-python'); // Find all <code> with class 'language-python'

    codeTags.forEach(codeTag => {
        const codeContent = codeTag.innerText.trim(); // Get the content inside the <code> block
        const preTag = codeTag.parentElement; // Get the parent <pre> element

        // Create a new textarea element
        const textarea = document.createElement('textarea');
        const randomId = 'cm-editor-' + Math.random().toString(36).substring(7); // Generate a random ID
        textarea.id = randomId;
        textarea.value = codeContent; // Set the value of the textarea to the Python code

        // Insert the textarea after the <pre> tag
        preTag.insertAdjacentElement('afterend', textarea);

        // Initialize CodeMirror from the newly created textarea
        CodeMirror.fromTextArea(textarea, {
            mode: "text/x-python",
            theme: "monokai",  // Apply the desired theme
            lineNumbers: true,
            matchBrackets: true,
            styleActiveLine: true,
            lineWrapping: true,
            tabSize: 4,
            indentUnit: 4,
            readOnly: true,  // Read-only editor
        });
    });
}

function initCodeMirrorForOutputTags() {

    const codeTags = document.querySelectorAll('code.text-output'); // Find all <code> with class 'language-python'

    codeTags.forEach(codeTag => {
        const codeContent = codeTag.innerText.trim(); // Get the content inside the <code> block
        const preTag = codeTag.parentElement; // Get the parent <pre> element

        // Create a new textarea element
        const textarea = document.createElement('textarea');
        const randomId = 'cm-editor-' + Math.random().toString(36).substring(7); // Generate a random ID
        textarea.id = randomId;
        textarea.value = codeContent; // Set the value of the textarea to the Python code

        // Insert the textarea after the <pre> tag
        preTag.insertAdjacentElement('afterend', textarea);

        // Initialize CodeMirror from the newly created textarea
        CodeMirror.fromTextArea(textarea, {
            mode: "text/plain",  // Plain text mode
            theme: "zenburn",    // Or any theme you prefer
            lineNumbers: false, // Disable line numbers for text output
            readOnly: true,     // Make the editor read-only
            lineWrapping: true, // Enable line wrapping
            styleActiveLine: false, // Disable active line styling
            showCursorWhenSelecting: false
        });
    });
}

function initCodeMirrorForJSONTags() {

    const codeTags = document.querySelectorAll('code.language-json'); // Find all <code> with class 'language-python'

    codeTags.forEach(codeTag => {
        const codeContent = codeTag.innerText.trim(); // Get the content inside the <code> block
        const preTag = codeTag.parentElement; // Get the parent <pre> element

        // Create a new textarea element
        const textarea = document.createElement('textarea');
        const randomId = 'cm-editor-' + Math.random().toString(36).substring(7); // Generate a random ID
        textarea.id = randomId;
        textarea.value = codeContent; // Set the value of the textarea to the Python code

        // Insert the textarea after the <pre> tag
        preTag.insertAdjacentElement('afterend', textarea);

        // Initialize CodeMirror from the newly created textarea
        CodeMirror.fromTextArea(textarea, {
            mode: "application/json",  // Plain text mode
            theme: "monokai",    // Or any theme you prefer
            lineNumbers: false, // Disable line numbers for text output
            readOnly: true,     // Make the editor read-only
            lineWrapping: true, // Enable line wrapping
            styleActiveLine: false, // Disable active line styling
            showCursorWhenSelecting: false
        });
    });
}

document.addEventListener('DOMContentLoaded', () => {
    // Call the function to initialize CodeMirror
    initCodeMirrorForCodeTags();
    initCodeMirrorForOutputTags();
    initCodeMirrorForJSONTags();
})
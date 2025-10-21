import os
import mimetypes
from urllib.parse import urlparse
import urllib.request
import base64


class Media:
    """
    A utility class for handling media files and URLs.

    This class provides static methods for working with media files, including
    converting them to formats suitable for API consumption, determining file types,
    and handling both local files and URLs.
    """

    @staticmethod
    def get_as_openai_dict(path: str) -> dict:
        """
        Convert a media file to a dictionary format suitable for OpenAI API.

        Parameters
        ----------
        path : str
            The path to the media file or URL.

        Returns
        -------
        dict
            A dictionary containing the media data in OpenAI-compatible format.
            For images, returns a dictionary with type "image_url" and base64-encoded data.
            For audio, returns a dictionary with type "input_audio" and base64-encoded data.

        Raises
        ------
        ValueError
            If the media type is not supported or if the file cannot be processed.
        """

        main_type, file_mime = Media._get_file_type(path)

        if main_type == "image":
            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{file_mime};base64,{Media.path_to_base64(path)}",
                }}
        elif main_type == "audio":
            return {
                "type": "input_audio",
                "input_audio": {
                    "data": Media.path_to_base64(path),
                    "format": file_mime
            }}
        else:
            raise ValueError(f"Unsupported media type: {main_type}")

    @staticmethod
    def _get_file_type(file_path: str) -> tuple[str, str]:
        """
        Determine the file type based on the file extension.

        Parameters
        ----------
        file_path : str
            The path to the file to analyze.

        Returns
        -------
        tuple[str, str]
            A tuple containing (main_type, subtype) where:
            - main_type is the primary media type (e.g., 'image', 'audio')
            - subtype is the specific format (e.g., 'jpeg', 'mp3')

        Raises
        ------
        ValueError
            If the file type cannot be determined or is not supported.
        """
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type:
            main_type, subtype = mime_type.split('/')
            return main_type, subtype  # Returns both the main type and subtype
        raise ValueError(f"Unsupported file type: {file_path}")


    @staticmethod
    def path_to_base64(path: str) -> str:
        """
        Convert the target's content of a file OR URL to a base64-encoded string.

        Parameters
        ----------
        path : str
            The path to the file or URL to convert.

        Returns
        -------
        str
            The base64-encoded content of the file or URL.

        Raises
        ------
        ValueError
            If the path is neither a valid URL nor a file on disk, or if there's
            an error fetching the URL content.
        """
        if Media.is_url(path):
            try:
                with urllib.request.urlopen(path) as response:
                    data = response.read()
            except Exception as e:
                raise ValueError(f"Failed to fetch URL: {e}")
        elif os.path.isfile(path):
            with open(path, "rb") as file:
                data = file.read()
        else:
            raise ValueError(f"Invalid path: {path} is neither a valid URL nor a file on disk.")

        # Convert to base64
        return base64.b64encode(data).decode("utf-8")

    @staticmethod
    def is_url(path: str) -> bool:
        """
        Check if the given path is a valid URL.

        Parameters
        ----------
        path : str
            The path to check.

        Returns
        -------
        bool
            True if the path is a valid URL (has both scheme and netloc),
            False otherwise.
        """
        parsed = urlparse(path)
        return bool(parsed.scheme and parsed.netloc)
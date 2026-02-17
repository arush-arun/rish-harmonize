"""
MRtrix3 I/O utilities.
"""

import subprocess
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import json


class MRtrixIO:
    """Utilities for reading/writing MRtrix3 files."""
    
    @staticmethod
    def get_info(image: str) -> Dict:
        """
        Get image info using mrinfo.
        
        Parameters
        ----------
        image : str
            Path to image
            
        Returns
        -------
        dict
            Image information
        """
        result = subprocess.run(
            ["mrinfo", "-json_all", image],
            capture_output=True,
            text=True,
            check=True
        )
        
        return json.loads(result.stdout)
    
    @staticmethod
    def get_size(image: str) -> Tuple[int, ...]:
        """Get image dimensions."""
        result = subprocess.run(
            ["mrinfo", "-size", image],
            capture_output=True,
            text=True,
            check=True
        )
        return tuple(int(x) for x in result.stdout.strip().split())
    
    @staticmethod
    def get_voxel_size(image: str) -> Tuple[float, ...]:
        """Get voxel dimensions."""
        result = subprocess.run(
            ["mrinfo", "-spacing", image],
            capture_output=True,
            text=True,
            check=True
        )
        return tuple(float(x) for x in result.stdout.strip().split())
    
    @staticmethod
    def convert(
        input_image: str,
        output_image: str,
        **kwargs
    ) -> str:
        """
        Convert image format.
        
        Parameters
        ----------
        input_image : str
            Input image path
        output_image : str
            Output image path
        **kwargs
            Additional mrconvert options
            
        Returns
        -------
        str
            Output path
        """
        cmd = ["mrconvert", input_image, output_image, "-force"]
        
        for key, val in kwargs.items():
            cmd.extend([f"-{key}", str(val)])
        
        subprocess.run(cmd, check=True)
        return output_image
    
    @staticmethod
    def extract_volumes(
        image: str,
        output: str,
        volumes: List[int]
    ) -> str:
        """
        Extract specific volumes from 4D image.
        
        Parameters
        ----------
        image : str
            Input 4D image
        output : str
            Output image
        volumes : list of int
            Volume indices to extract
            
        Returns
        -------
        str
            Output path
        """
        vol_str = ",".join(str(v) for v in volumes)
        subprocess.run([
            "mrconvert", image,
            "-coord", "3", vol_str,
            output, "-force"
        ], check=True)
        
        return output
    
    @staticmethod
    def math_operation(
        images: List[str],
        operation: str,
        output: str,
        axis: Optional[int] = None
    ) -> str:
        """
        Perform math operation on images.
        
        Parameters
        ----------
        images : list of str
            Input images
        operation : str
            Operation (mean, sum, min, max, etc.)
        output : str
            Output image
        axis : int, optional
            Axis for operation (for single image)
            
        Returns
        -------
        str
            Output path
        """
        cmd = ["mrmath", *images, operation, output, "-force"]
        
        if axis is not None:
            cmd.extend(["-axis", str(axis)])
        
        subprocess.run(cmd, check=True)
        return output
    
    @staticmethod
    def check_mrtrix_installed() -> bool:
        """Check if MRtrix3 is installed."""
        try:
            result = subprocess.run(
                ["mrinfo", "--version"],
                capture_output=True,
                check=True
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    @staticmethod
    def get_mrtrix_version() -> Optional[str]:
        """Get MRtrix3 version."""
        try:
            result = subprocess.run(
                ["mrinfo", "--version"],
                capture_output=True,
                text=True,
                check=True
            )
            # Parse version from output
            for line in result.stdout.split("\n"):
                if "mrinfo" in line.lower():
                    return line.strip()
            return result.stdout.strip().split("\n")[0]
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

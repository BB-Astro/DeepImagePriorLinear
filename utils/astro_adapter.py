"""
Astronomical Image Adapter for Deep Image Prior
Handles 16-bit TIFF/FITS images with proper dynamic range preservation

Author: Claude & Ben
Date: 2025-11-09
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
import warnings

# Import based on availability
try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False
    warnings.warn("tifffile not installed. TIFF 16-bit support will be limited.")

try:
    from astropy.io import fits
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False
    # Silently ignore - FITS is optional

try:
    import xisf
    HAS_XISF = True
except ImportError:
    HAS_XISF = False
    # Silently ignore - XISF is optional


class AstroImageHandler:
    """
    Specialized handler for astronomical images (16-bit TIFF/FITS)

    Key features:
    - Preserves 16-bit dynamic range
    - Uses percentiles instead of min/max normalization
    - Maintains metadata (FITS headers, etc.)
    - Handles both TIFF and FITS formats
    """

    @staticmethod
    def _check_finite(data: np.ndarray, path: Path) -> None:
        """Fail fast on non-finite pixels: a single NaN propagates through
        the percentile normalization and turns the whole image into NaN,
        then poisons the loss and the network weights. Better to refuse the
        file with a precise diagnostic than to waste a multi-hour run."""
        finite = np.isfinite(data)
        n_bad = int(data.size - finite.sum())
        if n_bad:
            bad_idx = np.argwhere(~finite)
            first = tuple(int(v) for v in bad_idx[0])
            raise ValueError(
                f"{path.name}: {n_bad} non-finite pixel(s) (NaN/Inf), first at "
                f"index {first}. Clean or crop the image before denoising "
                f"(NaN borders are common in resampled/mosaicked data).")

    @staticmethod
    def _normalize_axes(data: np.ndarray, path: Path) -> np.ndarray:
        """Return data as (H, W) or (H, W, C). FITS cubes are commonly
        (C, H, W): transpose when the leading axis looks like channels and
        the trailing one does not. Ambiguous cubes are rejected."""
        if data.ndim == 2:
            return data
        if data.ndim != 3:
            raise ValueError(f"{path.name}: unsupported {data.ndim}D data")
        lead_ch = data.shape[0] in (1, 3)
        trail_ch = data.shape[2] in (1, 3)
        if trail_ch:
            return data
        if lead_ch:
            return np.ascontiguousarray(np.transpose(data, (1, 2, 0)))
        raise ValueError(
            f"{path.name}: ambiguous 3D shape {data.shape}, cannot identify "
            f"the channel axis. Split the channels and process each "
            f"separately.")

    @staticmethod
    def load_astro_image(
        path: Union[str, Path],
        preserve_range: bool = True,
        percentiles: Tuple[float, float] = (0.5, 99.5),
        target_channels: int = 3,
        norm_range: Optional[Tuple[float, float]] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Load astronomical image with dynamic range preservation

        Args:
            path: Path to image file (TIFF or FITS)
            preserve_range: Whether to preserve original dynamic range
            percentiles: Percentiles for normalization (default: 0.5%, 99.5%)
            target_channels: Convert to this many channels (1 or 3)
            norm_range: Absolute (low, high) normalization window in file
                units. When given, percentiles are NOT computed: the window is
                applied as-is. This is how tiled processing keeps every tile on
                the same dynamic range (window computed once on the full image).

        Returns:
            Tuple of (normalized_data [0,1], metadata_dict)
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        # Load based on format
        if path.suffix.lower() in ['.tif', '.tiff']:
            data, metadata = AstroImageHandler._load_tiff(path)
        elif path.suffix.lower() in ['.fit', '.fits']:
            if not HAS_ASTROPY:
                raise ImportError("astropy required for FITS files. Install with: pip install astropy")
            data, metadata = AstroImageHandler._load_fits(path)
        elif path.suffix.lower() in ['.xisf']:
            if not HAS_XISF:
                raise ImportError("xisf required for XISF files. Install with: pip install xisf")
            data, metadata = AstroImageHandler._load_xisf(path)
        else:
            raise ValueError(f"Unsupported format: {path.suffix}. Supported: .tif, .tiff, .fit, .fits, .xisf")

        print(f"📷 Loaded {path.name}: shape={data.shape}, dtype={data.dtype}")

        # Ensure float32, finite, channels-last
        data = data.astype(np.float32)
        AstroImageHandler._check_finite(data, path)
        data = AstroImageHandler._normalize_axes(data, path)

        # Collapse channels BEFORE the statistics so that the percentile
        # window and the reinjection mask live in the same geometry as the
        # saved output (the pipeline is monochrome; distinct RGB channels
        # are averaged with a warning when mono is requested).
        if data.ndim == 3:
            if data.shape[2] == 1:
                data = data[..., 0]
            elif all(np.array_equal(data[..., 0], data[..., c])
                     for c in range(1, data.shape[2])):
                data = data[..., 0]
                print("   Identical channels collapsed to mono")
            elif target_channels == 1:
                warnings.warn(f"{path.name}: distinct RGB channels averaged "
                              f"to mono; process channels separately to keep "
                              f"color information")
                data = data.mean(axis=2)

        print(f"   Raw range: [{data.min():.6g}, {data.max():.6g}]")

        # Normalize with percentiles to preserve dynamic range
        if preserve_range:
            if norm_range is not None:
                p_low, p_high = float(norm_range[0]), float(norm_range[1])
                print(f"   Using imposed normalization window: [{p_low:.6g}, {p_high:.6g}]")
            else:
                p_low, p_high = np.percentile(data, percentiles)
                print(f"   Using percentiles {percentiles}: [{p_low:.6g}, {p_high:.6g}]")

            # Avoid division by zero
            if p_high - p_low < 1e-6:
                p_low = data.min()
                p_high = data.max()
                print(f"   ⚠️  Percentiles too close, using min/max: [{p_low:.6g}, {p_high:.6g}]")

            data_norm = np.clip((data - p_low) / (p_high - p_low), 0, 1)

            # Pixels above the high percentile (star cores) are flattened to 1.0
            # by the clip: the network never sees their true values, so they must
            # be written back verbatim at save time to keep the photometry exact.
            # The low side is NOT kept: dark outliers are noise excursions, and
            # replacing them with the denoised background estimate is the point.
            high_mask = data > p_high
            metadata['clip_high_mask'] = high_mask
            metadata['clip_high_values'] = data[high_mask]
            metadata['clip_high_count'] = int(high_mask.sum())
            if metadata['clip_high_count'] > 0:
                print(f"   Kept {metadata['clip_high_count']} pixels above p{percentiles[1]} "
                      f"for reinjection at save time")

            # Store original range in metadata
            metadata['original_min'] = p_low
            metadata['original_max'] = p_high
            metadata['percentiles_used'] = percentiles
        else:
            # Simple normalization
            if data.max() > 1:
                data_norm = data / 65535.0  # Assume 16-bit
            else:
                data_norm = data

        # Handle channel conversion
        if len(data_norm.shape) == 2:
            # Grayscale image
            if target_channels == 3:
                # Convert to RGB by duplicating channels
                data_norm = np.stack([data_norm] * 3, axis=2)
                print(f"   Converted grayscale to RGB: {data_norm.shape}")
            elif target_channels == 1:
                # Keep as grayscale, add channel dimension
                data_norm = data_norm[..., np.newaxis]
        elif len(data_norm.shape) == 3:
            # Already has channels
            if data_norm.shape[2] == 1 and target_channels == 3:
                # Single channel to RGB
                data_norm = np.concatenate([data_norm] * 3, axis=2)
            elif data_norm.shape[2] == 3 and target_channels == 1:
                # RGB to grayscale
                data_norm = np.mean(data_norm, axis=2, keepdims=True)

        print(f"   Normalized range: [{data_norm.min():.3f}, {data_norm.max():.3f}]")
        print(f"   Final shape: {data_norm.shape}")

        return data_norm, metadata

    @staticmethod
    def load_raw(path: Union[str, Path]) -> Tuple[np.ndarray, Dict]:
        """Load the raw 2D data in file units, without any normalization.

        Multi-channel data is accepted only when all channels are identical
        (collapsed to one); truly distinct channels must be processed
        separately by the caller.
        """
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix in ('.fit', '.fits'):
            if not HAS_ASTROPY:
                raise ImportError("astropy required for FITS files")
            data, metadata = AstroImageHandler._load_fits(path)
        elif suffix in ('.tif', '.tiff'):
            data, metadata = AstroImageHandler._load_tiff(path)
        elif suffix == '.xisf':
            if not HAS_XISF:
                raise ImportError("xisf required for XISF files")
            data, metadata = AstroImageHandler._load_xisf(path)
        else:
            raise ValueError(f"Unsupported format: {suffix}")

        data = np.asarray(data, dtype=np.float32)
        AstroImageHandler._check_finite(data, path)
        data = AstroImageHandler._normalize_axes(data, path)
        if data.ndim == 3:
            if data.shape[2] == 1:
                data = data[..., 0]
            elif all(np.array_equal(data[..., 0], data[..., c])
                     for c in range(1, data.shape[2])):
                data = data[..., 0]
            else:
                raise ValueError(
                    "load_raw handles one channel at a time. "
                    "Split the channels and process each separately.")
        return data, metadata

    @staticmethod
    def _load_tiff(path: Path) -> Tuple[np.ndarray, Dict]:
        """Load TIFF file with tifffile or PIL fallback"""
        metadata = {'format': 'TIFF', 'source_path': str(path)}

        if HAS_TIFFFILE:
            # Preferred: tifffile for proper 16-bit support
            data = tifffile.imread(path)

            # Try to get metadata if available
            try:
                with tifffile.TiffFile(path) as tif:
                    if hasattr(tif, 'pages') and len(tif.pages) > 0:
                        page = tif.pages[0]
                        if hasattr(page, 'tags'):
                            metadata['tiff_tags'] = {
                                tag.name: tag.value
                                for tag in page.tags.values()
                                if hasattr(tag, 'name')
                            }
            except:
                pass  # Metadata is optional
        else:
            # Fallback: PIL (may have issues with 16-bit)
            from PIL import Image
            img = Image.open(path)
            data = np.array(img)
            if img.mode == 'I;16':
                # PIL loaded as 16-bit signed, need to handle properly
                data = data.astype(np.uint16)

        return data, metadata

    @staticmethod
    def _load_fits(path: Path) -> Tuple[np.ndarray, Dict]:
        """Load FITS file with astropy"""
        with fits.open(path) as hdul:
            # Get primary HDU data
            data = hdul[0].data

            if data is None:
                # Try first extension
                if len(hdul) > 1:
                    data = hdul[1].data
                else:
                    raise ValueError(f"No data found in FITS file: {path}")

            # FITS data often comes in weird byte order, ensure native
            data = np.asarray(data, dtype=np.float32)

            # Store header as metadata
            metadata = {
                'format': 'FITS',
                'source_path': str(path),
                'header': dict(hdul[0].header)
            }

        return data, metadata

    @staticmethod
    def _load_xisf(path: Path) -> Tuple[np.ndarray, Dict]:
        """Load XISF file (PixInsight format)"""
        xisf_file = xisf.XISF(str(path))

        # Get metadata
        file_metadata = xisf_file.get_file_metadata()
        images_metadata = xisf_file.get_images_metadata()

        # Read first image
        data = xisf_file.read_image(0)

        # XISF uses channels-last (H, W, C), ensure we get (H, W) or (H, W, C)
        if len(data.shape) == 3 and data.shape[2] == 1:
            data = data.squeeze(axis=2)  # Remove single channel dimension

        # Convert to native byte order if needed
        data = np.asarray(data, dtype=np.float32)

        metadata = {
            'format': 'XISF',
            'source_path': str(path),
            'xisf_metadata': file_metadata,
            'image_metadata': images_metadata[0] if images_metadata else {}
        }

        return data, metadata

    @staticmethod
    def save_astro_image(
        data: np.ndarray,
        metadata: Dict,
        path: Union[str, Path],
        bit_depth: int = 16,
        compression: Optional[str] = None,
        reinject_clipped: bool = True
    ) -> None:
        """
        Save astronomical image with metadata preservation

        Args:
            data: Normalized image data [0, 1]
            metadata: Metadata dict with original_min/max
            path: Output path
            bit_depth: Output bit depth (8, 16, or 32)
            compression: Compression for TIFF (None, 'lzw', 'deflate')
            reinject_clipped: Write back the original values of pixels that
                were clipped above the high percentile at load time
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Remove channel dimension if grayscale
        if len(data.shape) == 3:
            if data.shape[2] == 1:
                data = data.squeeze(axis=2)
            elif data.shape[2] == 3 and np.allclose(data[..., 0], data[..., 1]) and np.allclose(data[..., 1], data[..., 2]):
                # RGB but all channels identical -> save as grayscale
                data = data[..., 0]
                print("   Converting identical RGB channels to grayscale for saving")

        # Restore original dynamic range if available
        if 'original_min' in metadata and 'original_max' in metadata:
            vmin = metadata['original_min']
            vmax = metadata['original_max']
            data_restored = data * (vmax - vmin) + vmin
            print(f"   Restoring dynamic range: [{vmin:.6g}, {vmax:.6g}]")
        elif bit_depth == 32:
            # Float32 output without a source range: keep [0, 1] as-is.
            # (bit_depth=32 means float storage, NOT a 32-bit integer scale.)
            data_restored = data.astype(np.float32)
            print("   No source range: float32 output kept in [0, 1]")
        else:
            # Scale to integer bit depth
            max_val = (2 ** bit_depth) - 1
            data_restored = data * max_val
            print(f"   Scaling to {bit_depth}-bit: [0, {max_val}]")

        # Reinject the pixels clipped above the high percentile at load time.
        # The percentile normalization flattened them to 1.0, the network only
        # ever saw that plateau: copying the original values back is the only
        # way to keep the photometry of bright star cores exact.
        metadata['reinjected_count'] = 0
        if reinject_clipped and metadata.get('clip_high_count', 0) > 0:
            mask = metadata['clip_high_mask']
            if mask.shape == data_restored.shape:
                data_restored = data_restored.copy()
                data_restored[mask] = metadata['clip_high_values']
                metadata['reinjected_count'] = metadata['clip_high_count']
                print(f"   Reinjected {metadata['clip_high_count']} clipped pixels "
                      f"(bright star cores, exact photometry)")
            else:
                print(f"   ⚠️ Clip mask shape {mask.shape} does not match data shape "
                      f"{data_restored.shape}: reinjection skipped")

        # Guard: integer output would quantize away a restored range that
        # spans less than 8 bits of levels. This happens for float32 sources
        # (XISF, float FITS) whose native range is far below 1 ADU: casting
        # to uint16 then yields an all-black file. Fall back to float32,
        # which both TIFF and FITS support.
        if bit_depth != 32:
            span = float(data_restored.max()) - float(data_restored.min())
            if span < 256:
                print(f"   ⚠️ Restored range spans only {span:.6g} levels: "
                      f"{bit_depth}-bit output would destroy it. Saving float32 instead.")
                bit_depth = 32

        # Clip and convert to appropriate dtype
        if bit_depth == 8:
            data_final = np.clip(data_restored, 0, 255).astype(np.uint8)
        elif bit_depth == 16:
            data_final = np.clip(data_restored, 0, 65535).astype(np.uint16)
        elif bit_depth == 32:
            data_final = data_restored.astype(np.float32)
        else:
            raise ValueError(f"Unsupported bit depth: {bit_depth}. Use 8, 16, or 32.")

        # Save based on format
        if path.suffix.lower() in ['.tif', '.tiff']:
            AstroImageHandler._save_tiff(data_final, path, compression, metadata)
        elif path.suffix.lower() in ['.fit', '.fits']:
            if not HAS_ASTROPY:
                # Convert to TIFF instead
                path = path.with_suffix('.tif')
                print(f"   ⚠️  astropy not available, saving as TIFF: {path}")
                AstroImageHandler._save_tiff(data_final, path, compression, metadata)
            else:
                AstroImageHandler._save_fits(data_final, path, metadata)
        elif path.suffix.lower() == '.xisf':
            if not HAS_XISF:
                path = path.with_suffix('.tif')
                print(f"   ⚠️  xisf not available, saving as TIFF: {path}")
                AstroImageHandler._save_tiff(data_final, path, compression, metadata)
            else:
                AstroImageHandler._save_xisf(data_final, path, metadata)
        else:
            # Default to TIFF
            path = path.with_suffix('.tif')
            AstroImageHandler._save_tiff(data_final, path, compression, metadata)

        print(f"💾 Saved: {path}")
        print(f"   Shape: {data_final.shape}, dtype: {data_final.dtype}")
        print(f"   Range: [{data_final.min()}, {data_final.max()}]")

    @staticmethod
    def _save_tiff(data: np.ndarray, path: Path, compression: Optional[str], metadata: Dict) -> None:
        """Save as TIFF with tifffile or PIL fallback"""
        if HAS_TIFFFILE:
            # Add metadata as ImageDescription
            description = f"Deep Image Prior Denoised\n"
            if 'percentiles_used' in metadata:
                description += f"Percentiles: {metadata['percentiles_used']}\n"
            if 'original_min' in metadata:
                description += f"Original range: [{metadata['original_min']:.6g}, {metadata['original_max']:.6g}]\n"

            tifffile.imwrite(
                path,
                data,
                compression=compression,
                description=description
            )
        else:
            # Fallback to PIL
            from PIL import Image

            if data.dtype == np.uint16:
                # PIL needs special mode for 16-bit
                img = Image.fromarray(data, mode='I')
            else:
                img = Image.fromarray(data)

            img.save(path, compression=compression)

    # FITS keywords that must NEVER be copied from a source header onto a
    # pipeline output. Structural keys are managed by astropy for the new
    # data; BSCALE/BZERO/BLANK describe the SOURCE integer encoding and,
    # copied onto a float32 output, would be re-applied at read time
    # (a uint16 source with BZERO=32768 would silently shift the whole
    # output by 32768 ADU). Checksums would simply be wrong. Pipeline keys
    # are excluded so a re-processed file cannot resurrect stale values.
    FITS_HEADER_EXCLUDE = frozenset({
        'SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'NAXIS3',
        'EXTEND', 'PCOUNT', 'GCOUNT',
        'BSCALE', 'BZERO', 'BLANK', 'CHECKSUM', 'DATASUM',
        'ORIGMIN', 'ORIGMAX', 'PERCLOW', 'PERCHIGH', 'NREINJ',
    })

    @staticmethod
    def _save_fits(data: np.ndarray, path: Path, metadata: Dict) -> None:
        """Save as FITS with astropy"""
        # Create HDU
        hdu = fits.PrimaryHDU(data)

        # Preserve the original header first (excluding structural, scaling
        # and pipeline keys), so that the pipeline cards written afterwards
        # always win over stale values from an earlier processing.
        if 'header' in metadata and isinstance(metadata['header'], dict):
            for key, value in metadata['header'].items():
                if str(key).upper() not in AstroImageHandler.FITS_HEADER_EXCLUDE:
                    try:
                        hdu.header[key] = value
                    except Exception:
                        pass  # Skip problematic headers

        # Add metadata to header
        hdu.header['HISTORY'] = 'Deep Image Prior denoising applied'

        if 'original_min' in metadata:
            hdu.header['ORIGMIN'] = (metadata['original_min'], 'Original minimum before normalization')
            hdu.header['ORIGMAX'] = (metadata['original_max'], 'Original maximum before normalization')

        if 'percentiles_used' in metadata:
            hdu.header['PERCLOW'] = (metadata['percentiles_used'][0], 'Low percentile used')
            hdu.header['PERCHIGH'] = (metadata['percentiles_used'][1], 'High percentile used')

        if 'reinjected_count' in metadata:
            hdu.header['NREINJ'] = (metadata['reinjected_count'],
                                    'Pixels above high percentile reinjected verbatim')

        # Write to file
        hdu.writeto(path, overwrite=True)

    @staticmethod
    def _save_xisf(data: np.ndarray, path: Path, metadata: Dict) -> None:
        """Save as XISF (PixInsight native format) with the xisf library"""
        # XISF stores channels-last: a mono image must be (H, W, 1)
        im = data[..., None] if data.ndim == 2 else data

        # Carry source metadata over. An XISF input provides the reader's
        # image_metadata structure directly; a FITS input provides a header
        # dict whose cards are replicated as FITSKeywords (same exclusions
        # as _save_fits, for the same reasons).
        fits_keywords: Dict = {}
        xisf_properties: Dict = {}
        src_meta = metadata.get('image_metadata')
        if isinstance(src_meta, dict):
            fits_keywords.update(src_meta.get('FITSKeywords', {}) or {})
            xisf_properties.update(src_meta.get('XISFProperties', {}) or {})
        elif isinstance(metadata.get('header'), dict):
            for key, value in metadata['header'].items():
                k = str(key).upper()
                if k and k not in AstroImageHandler.FITS_HEADER_EXCLUDE \
                        and k not in ('HISTORY', 'COMMENT', 'END'):
                    fits_keywords[str(key)] = [{'value': str(value), 'comment': ''}]

        def card(key: str, value, comment: str) -> None:
            fits_keywords[key] = [{'value': str(value), 'comment': comment}]

        if 'original_min' in metadata:
            card('ORIGMIN', metadata['original_min'], 'Original minimum before normalization')
            card('ORIGMAX', metadata['original_max'], 'Original maximum before normalization')
        if 'percentiles_used' in metadata:
            card('PERCLOW', metadata['percentiles_used'][0], 'Low percentile used')
            card('PERCHIGH', metadata['percentiles_used'][1], 'High percentile used')
        if 'reinjected_count' in metadata:
            card('NREINJ', metadata['reinjected_count'],
                 'Pixels above high percentile reinjected verbatim')

        xisf.XISF.write(
            str(path), im,
            creator_app='DIP_Astro_Silicon_MSE',
            image_metadata={'FITSKeywords': fits_keywords,
                            'XISFProperties': xisf_properties},
        )


def test_adapter():
    """Quick test of the adapter functionality"""
    print("🧪 Testing AstroImageHandler")

    # Test paths
    test_image = "deep_image_prior_astro_silicon/ImageTest/Arp204_cut.tif"

    if Path(test_image).exists():
        # Test loading
        print(f"\n📥 Testing load: {test_image}")
        data, metadata = AstroImageHandler.load_astro_image(test_image)
        print(f"✅ Loaded successfully")
        print(f"   Metadata keys: {list(metadata.keys())}")

        # Test saving
        output_path = "test_astro_adapter_output.tif"
        print(f"\n📤 Testing save: {output_path}")
        AstroImageHandler.save_astro_image(data, metadata, output_path)
        print(f"✅ Saved successfully")

        # Verify roundtrip
        data2, metadata2 = AstroImageHandler.load_astro_image(output_path)
        print(f"\n🔄 Roundtrip test:")
        print(f"   Data match: {np.allclose(data, data2, atol=1e-3)}")
        print(f"   Shape match: {data.shape == data2.shape}")

        # Clean up
        Path(output_path).unlink()
        print("\n✅ All tests passed!")
    else:
        print(f"❌ Test image not found: {test_image}")


if __name__ == "__main__":
    test_adapter()